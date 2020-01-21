from __future__ import division
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="size changed")
import sys 
import os
import time
import math
import random
import pickle
try:
    import Queue as queue
except ImportError:
    import queue
import threading
#import h5py
import json
import numpy as np
import tensorflow as tf
from termcolor import colored, cprint
from scipy.misc import imread, imresize
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow

from config_adv import config, loadDatasetConfig, parseArgs
from preprocess_adv import Preprocesser, bold, bcolored, writeline, writelist
from model_adv import MACnet


############################################# loggers #############################################

# Writes log header to file 
def logInit():
    with open(config.logFile(), "a+") as outFile:
        writeline(outFile, config.expName)
        headers = ["epoch", "trainAcc", "valAcc", "trainLoss", "valLoss"]
        if config.evalTrain:
            headers += ["evalTrainAcc", "evalTrainLoss"]
        if config.extra:
            if config.evalTrain:
                headers += ["thAcc", "thLoss"]
            headers += ["vhAcc", "vhLoss"]
        headers += ["time", "lr"]

        writelist(outFile, headers)
        # lr assumed to be last

# Writes log record to file 
def logRecord(epoch, epochTime, lr, trainRes, evalRes, extraEvalRes):
    with open(config.logFile(), "a+") as outFile:
        record = [epoch, trainRes["acc"], evalRes["val"]["acc"], trainRes["loss"], evalRes["val"]["loss"]]
        if config.evalTrain:
            record += [evalRes["evalTrain"]["acc"], evalRes["evalTrain"]["loss"]]
        if config.extra:
            if config.evalTrain:
                record += [extraEvalRes["evalTrain"]["acc"], extraEvalRes["evalTrain"]["loss"]]
            record += [extraEvalRes["val"]["acc"], extraEvalRes["val"]["loss"]]
        record += [epochTime, lr]

        writelist(outFile, record)

# Gets last logged epoch and learning rate
def lastLoggedEpoch():
    with open(config.logFile(), "r") as inFile:
        lastLine = list(inFile)[-1].split(",") 
    epoch = int(lastLine[0])
    lr = float(lastLine[-1])   
    return epoch, lr 

################################## printing, output and analysis ##################################

# Analysis by type
analysisQuestionLims = [(0,18),(19,float("inf"))]
analysisProgramLims = [(0,12),(13,float("inf"))]

toArity = lambda instance: instance["programSeq"][-1].split("_", 1)[0]
toType = lambda instance: instance["programSeq"][-1].split("_", 1)[1]

def fieldLenIsInRange(field):
    return lambda instance, group: \
        (len(instance[field]) >= group[0] and
        len(instance[field]) <= group[1])

# Groups instances based on a key
def grouperKey(toKey):
    def grouper(instances):
        res = defaultdict(list)
        for instance in instances:
            res[toKey(instnace)].append(instance)
        return res
    return grouper

# Groups instances according to their match to condition
def grouperCond(groups, isIn):
    def grouper(instances):
        res = {}
        for group in groups:
            res[group] = (instance for instance in instances if isIn(instance, group))
        return res
    return grouper 

groupers = {
    "questionLength": grouperCond(analysisQuestionLims, fieldLenIsInRange("questionSeq")),
    "programLength": grouperCond(analysisProgramLims, fieldLenIsInRange("programSeq")),
    "arity": grouperKey(toArity),
    "type": grouperKey(toType)
}

# Computes average
def avg(instances, field):
    if len(instances) == 0:
        return 0.0
    return sum(instances[field]) / len(instances)

# Prints analysis of questions loss and accuracy by their group 
def printAnalysis(res):
    if config.analysisType != "":
        print("Analysis by {type}".format(type = config.analysisType))
        groups = groupers[config.analysisType](res["preds"])
        for key in groups:
            instances = groups[key]
            avgLoss = avg(instances, "loss")
            avgAcc = avg(instances, "acc")
            num = len(instances)
            print("Group {key}: Loss: {loss}, Acc: {acc}, Num: {num}".format(key, avgLoss, avgAcc, num))

# Print results for a tier
def printTierResults(tierName, res, color):
    if res is None:
        return

    print("{tierName} Loss: {loss}, {tierName} accuracy: {acc}".format(tierName = tierName,
        loss = bcolored(res["loss"], color), 
        acc = bcolored(res["acc"], color)))
    
    printAnalysis(res)

# Prints dataset results (for several tiers)
def printDatasetResults(trainRes, evalRes, extraEvalRes):
    printTierResults("Training", trainRes, "magenta")
    printTierResults("Training EMA", evalRes["evalTrain"], "red")
    printTierResults("Validation", evalRes["val"], "cyan")
    printTierResults("Extra Training EMA", extraEvalRes["evalTrain"], "red")
    printTierResults("Extra Validation", extraEvalRes["val"], "cyan")    

# Writes predictions for several tiers
def writePreds(preprocessor, evalRes, extraEvalRes):
    preprocessor.writePreds(evalRes["evalTrain"], "evalTrain")
    preprocessor.writePreds(evalRes["val"], "val")
    preprocessor.writePreds(evalRes["test"], "test")
    preprocessor.writePreds(extraEvalRes["evalTrain"], "evalTrain", "H")
    preprocessor.writePreds(extraEvalRes["val"], "val", "H")
    preprocessor.writePreds(extraEvalRes["test"], "test", "H")

############################################# session #############################################
# Initializes TF session. Sets GPU memory configuration.
def setSession():
    sessionConfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    if config.allowGrowth:
        sessionConfig.gpu_options.allow_growth = True
    if config.maxMemory < 1.0:
        sessionConfig.gpu_options.per_process_gpu_memory_fraction = config.maxMemory
    return sessionConfig

############################################## savers #############################################
# Initializes savers (standard, optional exponential-moving-average and optional for subset of variables)
def setSavers(model,macvars):
    saver = tf.train.Saver(max_to_keep = config.weightsToKeep,var_list=macvars)

    subsetSaver = None
    if config.saveSubset:
        isRelevant = lambda var: any(s in var.name for s in config.varSubset)
        relevantVars = [var for var in tf.global_variables() if isRelevant(var)]
        subsetSaver = tf.train.Saver(relevantVars, max_to_keep = config.weightsToKeep, allow_empty = True)
    
    emaSaver = None
    if config.useEMA: 
        model.emaDictnew = {}
        for key in model.emaDict.keys():
             if 'resnet101' not in key:
                 model.emaDictnew[key] = model.emaDict[key]
        emaSaver = tf.train.Saver(model.emaDictnew, max_to_keep = config.weightsToKeep)

    return {
        "saver": saver,
        "subsetSaver": subsetSaver,
        "emaSaver": emaSaver
    }

################################### restore / initialize weights ##################################
# Restores weights of specified / last epoch if on restore mod.
# Otherwise, initializes weights.  
def loadWeights(sess, saver, init):
    if config.restoreEpoch > 0 or config.restore:
        # restore last epoch only if restoreEpoch isn't set
        if config.restoreEpoch == 0:
            # restore last logged epoch
            config.restoreEpoch, config.lr = lastLoggedEpoch()
        print(bcolored("Restoring epoch {} and lr {}".format(config.restoreEpoch, config.lr),"cyan"))
        print(bcolored("Restoring weights", "blue"))
        saver.restore(sess, config.weightsFile(config.restoreEpoch))
        epoch = config.restoreEpoch
    else:
        print(bcolored("Initializing weights", "blue"))
        sess.run(init)
        logInit()
        epoch = 0

    return epoch 

###################################### training / evaluation ######################################
# Chooses data to train on (main / extra) data. 
def chooseTrainingData(data):
    trainingData = data["main"]["train"]
    alterData = None

    if config.extra:
        if config.trainExtra:
            if config.extraVal:
                trainingData = data["extra"]["val"]
            else:
                trainingData = data["extra"]["train"]                  
        if config.alterExtra:
            alterData = data["extra"]["train"]

    return trainingData, alterData

#### evaluation
# Runs evaluation on train / val / test datasets.
def runEvaluation(sess, model, data, epoch, evalTrain = False, evalTest = False, getAtt = None):
    if getAtt is None:
        getAtt = config.getAtt
    res = {"evalTrain": None, "val": None, "test": None}
    
    if data is not None:
        if evalTrain and config.evalTrain:
            res["evalTrain"] = runEpoch(sess, model, data["evalTrain"], train = False, epoch = epoch, getAtt = getAtt)

        res["val"] = runEpoch(sess, model, data["val"], train = False, epoch = epoch, getAtt = getAtt)
        
        if evalTest or config.test:
            res["test"] = runEpoch(sess, model, data["test"], train = False, epoch = epoch, getAtt = getAtt)    
        
    return res

## training conditions (comparing current epoch result to prior ones)
def improveEnough(curr, prior, lr):
    prevRes = prior["prev"]["res"]
    currRes = curr["res"]

    if prevRes is None:
        return True

    prevTrainLoss = prevRes["train"]["loss"]
    currTrainLoss = currRes["train"]["loss"]
    lossDiff = prevTrainLoss - currTrainLoss
    
    notImprove = ((lossDiff < 0.015 and prevTrainLoss < 0.5 and lr > 0.00002) or \
                  (lossDiff < 0.008 and prevTrainLoss < 0.15 and lr > 0.00001) or \
                  (lossDiff < 0.003 and prevTrainLoss < 0.10 and lr > 0.000005))
                  #(prevTrainLoss < 0.2 and config.lr > 0.000015)
    
    return not notImprove

def better(currRes, bestRes):
    return currRes["val"]["acc"] > bestRes["val"]["acc"]

############################################## data ###############################################
#### instances and batching 
# Trims sequences based on their max length.
def trim2DVectors(vectors, vectorsLengths):
    maxLength = np.max(vectorsLengths)
    return vectors[:,:maxLength]

# Trims batch based on question length.
def trimData(data):
    data["questions"] = trim2DVectors(data["questions"], data["questionLengths"])
    return data

# Gets batch / bucket size.
def getLength(data):
    return len(data["instances"])

# Selects the data entries that match the indices. 
def selectIndices(data, indices):
    def select(field, indices): 
        if type(field) is np.ndarray:
            return field[indices]
        if type(field) is list:
            return [field[i] for i in indices]
        else:
            return field
    selected = {k : select(d, indices) for k,d in data.items()}
    return selected

# Selects the data entries that match the indices. 
def getindex(data, index):
    def select(field, index): 
        if type(field) is np.ndarray:
            return np.array([field[index]])
        if type(field) is list:
            return [field[index]]
        else:
            return field
    selected = {k : select(d, index) for k,d in data.items()}
    return selected

# Batches data into a a list of batches of batchSize. 
# Shuffles the data by default.
def getBatches(data, batchSize = None, shuffle = True):
    batches = []

    dataLen = getLength(data)
    if batchSize is None or batchSize > dataLen:
        batchSize = dataLen
    
    indices = np.arange(dataLen)
    if shuffle:
        np.random.shuffle(indices)

    for batchStart in range(0, dataLen, batchSize):
        batchIndices = indices[batchStart : batchStart + batchSize]
        # if len(batchIndices) == batchSize?
        if len(batchIndices) >= config.gpusNum:
            batch = selectIndices(data, batchIndices)
            batches.append(batch)
            # batchesIndices.append((data, batchIndices))

    return batches

#### image batches
# Opens image files.
def getimageandmask(image_path):
      img = imread(image_path, mode='RGB')
      edges = cv2.Canny(img,100,200)
      M = np.zeros(img.shape,dtype=np.uint8)
      xmin = max(np.min(np.where(edges!=0)[0]) - 2, 0)
      xmax = min(np.max(np.where(edges!=0)[0]) + 2,img.shape[0])
      ymin = max(np.min(np.where(edges!=0)[1]) - 2, 0)
      ymax = min(np.max(np.where(edges!=0)[1]) + 2,img.shape[1])
      M[xmin:xmax,ymin:ymax,:] = 1
      #M = 1 - M  # anti mask
      #M[:,:,:] = 1 # remove mask
      #img = img*M   # removing black mask
      img = imresize(img, (224,224), interp='bicubic')
      M = imresize(M, (224,224), interp='bicubic')
      img = img.astype(np.float32)
      M = M.astype(np.bool)
      return img, M

def openImageFiles(images):
    images["imagesFile"] = h5py.File(images["imagesFilename"], "r")
    images["imagesIds"] = None
    if config.dataset == "NLVR":
        with open(images["imageIdsFilename"], "r") as imageIdsFile:
            images["imagesIds"] = json.load(imageIdsFile)  

# Closes image files.
def closeImageFiles(images): 
    images["imagesFile"].close()

# Loads an images from file for a given data batch.
def loadImageBatch(images, batch):
    imagesFile = images["imagesFile"]
    id2idx = images["imagesIds"]

    toIndex = lambda imageId: imageId
    if id2idx is not None:
        toIndex = lambda imageId: id2idx[imageId]
    imageBatch = np.stack([imagesFile["features"][toIndex(imageId)] for imageId in batch["imageIds"]], axis = 0)
    
    return {"images": imageBatch, "imageIds": batch["imageIds"]}

# Loads images for several num batches in the batches list from start index. 
def loadImageBatches(images, batches, start, num):
    batches = batches[start: start + num]
    return [loadImageBatch(images, batch) for batch in batches]

#### data alternation
# Alternates main training batches with extra data.
def alternateData(batches, alterData, dataLen):
    alterData = alterData["data"][0] # data isn't bucketed for altered data

    # computes number of repetitions
    needed = math.ceil(len(batches) / config.alterNum) 
    print(bold("Extra batches needed: %d") % needed)
    perData = math.ceil(getLength(alterData) / config.batchSize)
    print(bold("Batches per extra data: %d") % perData)
    repetitions = math.ceil(needed / perData)
    print(bold("reps: %d") % repetitions)
    
    # make alternate batches
    alterBatches = []
    for _ in range(repetitions):
        repBatches = getBatches(alterData, batchSize = config.batchSize)
        random.shuffle(repBatches)
        alterBatches += repBatches
    print(bold("Batches num: %d") + len(alterBatches))
    
    # alternate data with extra data
    curr = len(batches) - 1
    for alterBatch in alterBatches:
        if curr < 0:
            # print(colored("too many" + str(curr) + " " + str(len(batches)),"red"))
            break
        batches.insert(curr, alterBatch)
        dataLen += getLength(alterBatch)
        curr -= config.alterNum

    return batches, dataLen

############################################ threading ############################################

imagesQueue = queue.Queue(maxsize = 20) # config.tasksNum
inQueue = queue.Queue(maxsize = 1)
outQueue = queue.Queue(maxsize = 1)

# Runs a worker thread(s) to load images while training .
class StoppableThread(threading.Thread):
    # Thread class with a stop() method. The thread itself has to check
    # regularly for the stopped() condition.

    def __init__(self, images, batches): # i
        super(StoppableThread, self).__init__()
        # self.i = i
        self.images = images
        self.batches = batches
        self._stop_event = threading.Event()

    # def __init__(self, args):
    #     super(StoppableThread, self).__init__(args = args)
    #     self._stop_event = threading.Event()

    # def __init__(self, target, args):
    #     super(StoppableThread, self).__init__(target = target, args = args)
    #     self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        while not self.stopped():
            try:
                batchNum = inQueue.get(timeout = 60)
                nextItem = loadImageBatches(self.images, self.batches, batchNum, int(config.taskSize / 2))
                outQueue.put(nextItem)
                # inQueue.task_done()
            except:
                pass
        # print("worker %d done", self.i)

def loaderRun(images, batches):
    batchNum = 0

    # if config.workers == 2:           
    #     worker = StoppableThread(images, batches) # i, 
    #     worker.daemon = True
    #     worker.start() 

    #     while batchNum < len(batches):
    #         inQueue.put(batchNum + int(config.taskSize / 2))
    #         nextItem1 = loadImageBatches(images, batches, batchNum, int(config.taskSize / 2))
    #         nextItem2 = outQueue.get()

    #         nextItem = nextItem1 + nextItem2
    #         assert len(nextItem) == min(config.taskSize, len(batches) - batchNum)
    #         batchNum += config.taskSize
            
    #         imagesQueue.put(nextItem)

    #     worker.stop()
    # else:
    while batchNum < len(batches):
        nextItem = loadImageBatches(images, batches, batchNum, config.taskSize)
        assert len(nextItem) == min(config.taskSize, len(batches) - batchNum)
        batchNum += config.taskSize                    
        imagesQueue.put(nextItem)

    # print("manager loader done")

########################################## stats tracking #########################################
# Computes exponential moving average.
def emaAvg(avg, value):
    if avg is None:
        return value
    emaRate = 0.98
    return avg * emaRate + value * (1 - emaRate)

# Initializes training statistics.
def initStats():
    return {
        "totalBatches": 0,
        "totalData": 0,
        "totalLoss": 0.0,
        "totalCorrect": 0,
        "loss": 0.0,
        "acc": 0.0,
        "emaLoss": None,
        "emaAcc": None,
    }

# Updates statistics with training results of a batch
def updateStats(stats, res, batch):
    stats["totalBatches"] += 1
    stats["totalData"] += getLength(batch)

    stats["totalLoss"] += res["loss"]
    stats["totalCorrect"] += res["correctNum"]

    stats["loss"] = stats["totalLoss"] / stats["totalBatches"]
    stats["acc"] = stats["totalCorrect"] / stats["totalData"]
    
    stats["emaLoss"] = emaAvg(stats["emaLoss"], res["loss"])
    stats["emaAcc"] = emaAvg(stats["emaAcc"], res["acc"])
                                                    
    return stats 

# auto-encoder ae = {:2.4f} autoEncLoss,
# Translates training statistics into a string to print
def statsToStr(stats, res, epoch, batchNum, dataLen, startTime):
    formatStr = "\reb {epoch},{batchNum} ({dataProcessed} / {dataLen:5d}), " + \
                             "t = {time} ({loadTime:2.2f}+{trainTime:2.2f}), " + \
                             "lr {lr}, l = {loss}, a = {acc}, avL = {avgLoss}, " + \
                             "avA = {avgAcc}, g = {gradNorm:2.4f}, " + \
                             "emL = {emaLoss:2.4f}, emA = {emaAcc:2.4f}; " + \
                             "{expname}" # {machine}/{gpu}"

    s_epoch = bcolored("{:2d}".format(epoch),"green")
    s_batchNum = "{:3d}".format(batchNum)
    s_dataProcessed = bcolored("{:5d}".format(stats["totalData"]),"green")
    s_dataLen = dataLen
    s_time = bcolored("{:2.2f}".format(time.time() - startTime),"green")
    s_loadTime = res["readTime"] 
    s_trainTime = res["trainTime"]
    s_lr = bold(config.lr)
    s_loss = bcolored("{:2.4f}".format(res["loss"]), "blue")
    s_acc = bcolored("{:2.4f}".format(res["acc"]),"blue")
    s_avgLoss = bcolored("{:2.4f}".format(stats["loss"]), "blue")
    s_avgAcc = bcolored("{:2.4f}".format(stats["acc"]),"red")
    s_gradNorm = res["gradNorm"]  
    s_emaLoss = stats["emaLoss"]
    s_emaAcc = stats["emaAcc"]
    s_expname = config.expName 
    # s_machine = bcolored(config.dataPath[9:11],"green") 
    # s_gpu = bcolored(config.gpus,"green")

    return formatStr.format(epoch = s_epoch, batchNum = s_batchNum, dataProcessed = s_dataProcessed,
                            dataLen = s_dataLen, time = s_time, loadTime = s_loadTime,
                            trainTime = s_trainTime, lr = s_lr, loss = s_loss, acc = s_acc,
                            avgLoss = s_avgLoss, avgAcc = s_avgAcc, gradNorm = s_gradNorm,
                            emaLoss = s_emaLoss, emaAcc = s_emaAcc, expname = s_expname)
                            # machine = s_machine, gpu = s_gpu)

# collectRuntimeStats, writer = None,  
'''
Runs an epoch with model and session over the data.
1. Batches the data and optionally mix it with the extra alterData.
2. Start worker threads to load images in parallel to training.
3. Runs model for each batch, and gets results (e.g. loss,  accuracy).
4. Updates and prints statistics based on batch results.
5. Once in a while (every config.saveEvery), save weights. 

Args:
    sess: TF session to run with.
    
    model: model to process data. Has runBatch method that process a given batch.
    (See model.py for further details).
    
    data: data to use for training/evaluation.
    
    epoch: epoch number.

    saver: TF saver to save weights

    calle: a method to call every number of iterations (config.calleEvery)

    alterData: extra data to mix with main data while training.

    getAtt: True to return model attentions.  
'''
def runEpoch(sess, model, data, train, epoch, saver = None, calle = None, 
    alterData = None, getAtt = False):
    # train = data["train"] better than outside argument

    # initialization
    startTime0 = time.time()

    stats = initStats()
    preds = []

    # open image files
    openImageFiles(data["images"])

    ## prepare batches
    buckets = data["data"]
    dataLen = sum(getLength(bucket) for bucket in buckets)
    
    # make batches and randomize
    batches = []
    for bucket in buckets:
        batches += getBatches(bucket, batchSize = config.batchSize)
    random.shuffle(batches)

    # alternate with extra data
    if train and alterData is not None:
        batches, dataLen = alternateData(batches, alterData, dataLen)

    # start image loaders
    if config.parallel:
        loader = threading.Thread(target = loaderRun, args = (data["images"], batches))
        loader.daemon = True
        loader.start()

    for batchNum, batch in enumerate(batches):   
        startTime = time.time()

        # prepare batch 
        batch = trimData(batch)

        # load images batch
        if config.parallel:
            if batchNum % config.taskSize == 0:
                imagesBatches = imagesQueue.get()
            imagesBatch = imagesBatches[batchNum % config.taskSize] # len(imagesBatches)     
        else:
            imagesBatch = loadImageBatch(data["images"], batch)
        for i, imageId in enumerate(batch["imageIds"]):
            assert imageId == imagesBatch["imageIds"][i]   
        
        # run batch
        res = model.runBatch(sess, batch, imagesBatch, train, getAtt) 

        # update stats
        stats = updateStats(stats, res, batch)
        preds += res["preds"]

        # if config.summerize and writer is not None:
        #     writer.add_summary(res["summary"], epoch)

        sys.stdout.write(statsToStr(stats, res, epoch, batchNum, dataLen, startTime))
        sys.stdout.flush()

        # save weights
        if saver is not None:
            if batchNum > 0 and batchNum % config.saveEvery == 0:
                print("")
                print(bold("saving weights"))
                saver.save(sess, config.weightsFile(epoch))

        # calle
        if calle is not None:            
            if batchNum > 0 and batchNum % config.calleEvery == 0:
                calle()
    
    sys.stdout.write("\r")
    sys.stdout.flush()

    print("")

    closeImageFiles(data["images"])

    if config.parallel:
        loader.join() # should work

    return {"loss": stats["loss"], 
            "acc": stats["acc"],
            "preds": preds
            }

def runattack(sess,adv_var, model, data, ans_lines, index, train,resnet_model,resnet_vars,saver, init, getAtt=False):
    max_iter = 1500 
    max_trials = 5
    #sess.run(tf.global_variables_initializer())
    #resnet_saver = tf.train.Saver(max_to_keep=None,var_list=resnet_vars)
    #resnet_saver.restore(sess, resnet_model)
    #sess.graph.finalize()
    #print("ResNet Loaded")
    # restore / initialize weights, initialize epoch variable
    #epoch = loadWeights(sess, saver, init)
    #print("MacNet Loaded")

    # get masked image and mask
    print("Path is:",data['imagePaths'][index])
    img, M = getimageandmask(data['imagePaths'][index])
    img = np.expand_dims(img,0)
    M = np.expand_dims(M,0)
    # run batch
    batch = getindex(data,index)
    print(batch['instances'][0][u'question'],batch['instances'][0][u'answer'])
    #feed = model.runBatch(sess, batch, img, M, train, getAtt)
    #print("Current answer:", ans_lines[np.argmax(feed["logits"][0])])
    #print("Type of attention maps",type(feed["attmaps"]))
    #print("keys of attention maps",feed["attmaps"].keys())
    #print(len(feed["attmaps"]['kb']), type(feed["attmaps"]['kb']))
    grad_scale = [100.0]
    success = []
    adv_image_j = []
    adv_iter_j = [] 
    score_val_j = []
    l2_distortion = []
    img_att_j = []
    init_img_att_j = []
    question_att_j = []
    adv_image_incorrect_j = []
    score_val_incorrect_j = []
    for j in range(max_trials):
        #sess.run(tf.initialize_variables([adv_var]))
        sess.run(tf.variables_initializer([adv_var]))
        batch['grad_scaling'] = grad_scale[j]
        print("Grad scaling is:",batch['grad_scaling'])
        for n_iter in range(max_iter):
            feed = model.runBatch(sess, batch, img, M, train, getAtt)
            if feed["acc"] == 1.0:
               success += [True]
               print("Successful %d"%(index))
               if any(not _ for _ in success):
                    last_false = len(success) - success[::-1].index(False) - 1
                    grad_scale += [0.5 * (grad_scale[j] + grad_scale[last_false])]
               else:
                    grad_scale+= [grad_scale[j] * 0.5]
               l2_distortion.append(np.linalg.norm(feed["advImg"] - img))
               adv_image_j.append(feed["advImg"])
               adv_iter_j.append(n_iter)
               score_val_j.append(feed["logits"])
               img_att_j.append(feed["attmaps"]['kb'])
               question_att_j.append(feed["attmaps"]['question'])
               break
            elif n_iter==max_iter-1:
               success += [False]
               print("Unsuccessful %d"%(index))
               if any(_ for _ in success):
                    last_true = len(success) - success[::-1].index(True) - 1
                    grad_scale += [0.5 * (grad_scale[j] + grad_scale[last_true])]
               else:
                    grad_scale += [grad_scale[j] * 2]
               adv_image_incorrect_j.append(feed["advImg"])
               score_val_incorrect_j.append(feed["logits"])
            

    #feed = model.runBatch(sess, batch, img, M, train, getAtt)
    #print(feed["advImg"].shape,np.max(feed["advImg"]),np.min(feed["advImg"]))
    #print(feed["imagefeature"].shape)
    #arr = np.squeeze(feed["attmaps"]['kb'])
    #print("Final non-zero attention values:",[(arr[i]!=0).sum() for i in range(4)])
    #arr = np.squeeze(feed["probmaps"])
    #print("Final non-zero attention values:",(arr!=0).sum())
    return success, grad_scale, adv_image_j, adv_iter_j,score_val_j,l2_distortion,img_att_j,init_img_att_j,question_att_j, adv_image_incorrect_j, score_val_incorrect_j


'''
Trains/evaluates the model:
1. Set GPU configurations.
2. Preprocess data: reads from datasets, and convert into numpy arrays.
3. Builds the TF computational graph for the MAC model.
4. Starts a session and initialize / restores weights.
5. If config.train is True, trains the model for number of epochs:
    a. Trains the model on training data
    b. Evaluates the model on training / validation data, optionally with 
       exponential-moving-average weights.
    c. Prints and logs statistics, and optionally saves model predictions.
    d. Optionally reduces learning rate if losses / accuracies don't improve,
       and applies early stopping.
6. If config.test is True, runs a final evaluation on the dataset and print
   final results!
'''
def main():
    with open(config.configFile(), "a+") as outFile:
        json.dump(vars(config), outFile)

    # set gpus
    if config.gpus != "":
        config.gpusNum = len(config.gpus.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    tf.logging.set_verbosity(tf.logging.ERROR)

    # process data
    print(bold("Preprocess data from adversarial..."))
    start = time.time()
    preprocessor = Preprocesser()
    data, embeddings, answerDict = preprocessor.preprocessData()

    #data = pickle.load(open("data.pkl"))
    #answerDict = pickle.load(open("answerDict.pkl"))
    #embeddings = np.load("embeddings.npy",allow_pickle=True)[()]
    config.answerWordsNum = answerDict.getNumSymbols()
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))

    print(data["main"]["val"]["data"][0]['imagePaths'][0])
    print(data["main"]["val"]["data"][0].keys())
    print(data["main"]["val"]["data"][0]['questions'][0])
    print(data["main"]["val"]["data"][0]['answers'][0])
    print(len(data["main"]["val"]["data"][0]['answers']))
    data_dict = data["main"]["val"]["data"][0]
    del(data)
    # getting answers
    ans_lines = open("answers_clevr.txt","r").readlines()
    for i in range(len(ans_lines)):
        ans_lines[i] = ans_lines[i].split("\n")[0]
    # build model
    print(bold("Building model..."))
    start = time.time()
    model = MACnet(embeddings, answerDict)
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))
    resnet_model = './resnet101_tf.ckpt' 
    reader = pywrap_tensorflow.NewCheckpointReader(resnet_model)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # initializer
    init = tf.global_variables_initializer()
    
    #print(tf.trainable_variables())
    resnet_vars = []
    mac_vars = []
    adv_var = tf.trainable_variables()[0]
    for v in tf.global_variables():
        if 'resnet101' in v.name and 'adv_weights' not in v.name and v.name.replace(":0","") in var_to_shape_map.keys():
            resnet_vars.append(v)

    macnet_model = 'weights/clevrExperiment/weights25.ckpt'
    reader1 = pywrap_tensorflow.NewCheckpointReader(macnet_model)
    var_to_shape_map1 = reader1.get_variable_to_shape_map()

    for v in tf.global_variables():
        if 'resnet101' not in v.name and v.name.replace(":0","") in var_to_shape_map1.keys():
            mac_vars.append(v)
    
    #for v in tf.trainable_variables():
    #     if 'resnet101' not in v.name:
    #         mac_vars.append(v)

    names = [k.name for k in resnet_vars]
    #print(names)
    print("Checking:",'macModel/resnet101/block3/conv_block/batch_normalization_2/moving_mean:0' in names)

    mac_names = [k.name for k in mac_vars]
    print("Checking for MacNet:", 'train/macModel/encoder/birnnLayer/bidirectional_rnn/bw/basic_lstm_cell/kernel/ExponentialMovingAverage:0' in mac_names)
    for key in var_to_shape_map:
         if key +":0" not in names:
             print("Key not found so far:",key)
             #resnet_vars.append(key)
    #print(resnet_vars[0].name)
    # savers
    savers = setSavers(model,mac_vars)
    saver, emaSaver = savers["saver"], savers["emaSaver"]

    # sessionConfig
    sessionConfig = setSession()
    with tf.Session(config = sessionConfig) as sess:
        sess.run(tf.global_variables_initializer())
        resnet_saver = tf.train.Saver(max_to_keep=None,var_list=resnet_vars)
        resnet_saver.restore(sess, resnet_model)
        print("ResNet Loaded")
        # restore / initialize weights, initialize epoch variable
        epoch = loadWeights(sess, saver, init)
        # NEW: loading emaSaver
        emaSaver.restore(sess, config.weightsFile(epoch))
        print("MacNet Loaded")

        if config.attack:
            correct = 0
            print("Starting attack")
            total_success = []
            grad_scaling_list = []
            adv_image_array = []
            pred = []
            adv_iters = []
            l2_norms = []
            cor_enum = []
            adv_image_incorrect_array = []
            pred_incorrect = []
            image_attention = []
            init_image_attention = []
            question_attention = []
            for i in range(100):
                success, grad_scale, adv_image_j, adv_iter_j, score_val_j, l2_distortion, img_att_j, init_img_att_j, question_att_j, adv_image_incorrect_j, score_val_incorrect_j  = runattack(sess, adv_var, model, data_dict,ans_lines, index=i, train = True, resnet_model=resnet_model,resnet_vars=resnet_vars,saver=saver,init=init)
                total_success.append(success)
                grad_scaling_list.append(grad_scale)
                init_image_attention.append(init_img_att_j)
                if any(_ for _ in success):
                    correct += 1
                    print(i+1, correct)
                    cor_enum.append(i)
                    adv_image_array.append(adv_image_j[np.argmin(l2_distortion)])
                    pred.append(score_val_j[np.argmin(l2_distortion)])
                    adv_iters.append(adv_iter_j[np.argmin(l2_distortion)])
                    l2_norms.append(np.min(l2_distortion))
                    image_attention.append(img_att_j[np.argmin(l2_distortion)])
                    question_attention.append(np.array(question_att_j[np.argmin(l2_distortion)]))
                elif not any(_ for _ in success):
                    adv_image_incorrect_array.append(adv_image_incorrect_j[-1])
                    pred_incorrect.append(score_val_incorrect_j[-1])

                if i%20 == 0 or i==99:
                    os.makedirs("./adv_clevr_dc_wb/%d_iterations"%(i+1))
                    np.save("./adv_clevr_dc_wb/%d_iterations/correct_ind.npy"%(i+1),np.array(cor_enum))
                    np.save("./adv_clevr_dc_wb/%d_iterations/success.npy"%(i+1),np.array(total_success))
                    np.save("./adv_clevr_dc_wb/%d_iterations/grad_scaling.npy"%(i+1),np.array(grad_scaling_list))
                    np.save("./adv_clevr_dc_wb/%d_iterations/adv_image_array.npy"%(i+1),np.array(adv_image_array))
                    np.save("./adv_clevr_dc_wb/%d_iterations/pred.npy"%(i+1),np.array(pred))
                    np.save("./adv_clevr_dc_wb/%d_iterations/iters.npy"%(i+1),np.array(adv_iters))
                    np.save("./adv_clevr_dc_wb/%d_iterations/l2_norms.npy"%(i+1),np.array(l2_norms))
                    np.save("./adv_clevr_dc_wb/%d_iterations/image_att.npy"%(i+1),np.array(image_attention))
                    np.save("./adv_clevr_dc_wb/%d_iterations/init_image_att.npy"%(i+1),np.array(init_image_attention))
                    #np.save("./adv_clevr/%d_iterations/quest_att.npy"%(i+1),np.array(question_attention))
                    np.save("./adv_clevr_dc_wb/%d_iterations/adv_image_incorrect_array.npy"%(i+1),np.array(adv_image_incorrect_array))
                    np.save("./adv_clevr_dc_wb/%d_iterations/pred_incorrect.npy"%(i+1),np.array(pred_incorrect))
                    pickle.dump(question_attention,open("./adv_clevr_dc_wb/%d_iterations/quest_att.pkl"%(i+1),"w"))
                    total_success = []
                    grad_scaling_list = []
                    adv_image_array = []
                    pred = []
                    adv_iters = []
                    l2_norms = []
                    cor_enum = []
                    adv_image_incorrect_array = []
                    pred_incorrect = []
                    image_attention = []
                    init_image_attention = []
                    question_attention = []

        #snapshot_saver = tf.train.Saver()
        #snapshot_saver.save(sess, "./trial.ckpt")
        #print(bcolored("Done!","white"))

if __name__ == '__main__':
    parseArgs()    
    loadDatasetConfig[config.dataset]()        
    main()


