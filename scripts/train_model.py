import sys
root = "C:\\Users\\saman\\Documents\\GitHub\\neural_seq_decoder\\" ##you will need to change this for operating on your system -Samantha
sys.path.append(root)


modelName = 'softDTW_debug' #note that the params below are from shreeram.py file on bruinlearn -Samantha


args = {}
args['outputDir'] = root + 'outputs\\' + modelName
args['datasetPath'] = "C:\\Users\\saman\\Documents\\Classes\\ECE 243A Brain Computer Interfaces\\Final Project\\competitionData\\decoder_dataset.pkl" ##you will need to change this for operating on your system
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 32 #I've had OOM issues with batch sizes bigger than 32 (ie: 64, 128), so increase at your own peril -Samantha
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

#Loss Function Parameters
args['lambda'] = 1.0 #Strength of regularization penalty for predicting sequences less than the target sequence length
args['gamma'] = 0.1 #strength of softmin operator
args['regTemp'] = 0.1 #Temperature of regularization penalty (less than one helps make the regularization penalty more accurate at the start of training)
args['maxPenalty'] = 5.0 #Upper bound on the negative log loss for the deletion path and the substitution/match path for the soft levenshtein loss

from src.neural_decoder.neural_decoder_trainer_softdtw import trainModel

trainModel(args)