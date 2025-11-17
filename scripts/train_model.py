import sys
root = "C:\\Users\\saman\\Documents\\GitHub\\neural_seq_decoder\\"
sys.path.append(root)


modelName = 'speechBaseline4'

args = {}
args['outputDir'] = 'C:\\Users\\saman\\Documents\\Classes\\ECE 243A Brain Computer Interfaces\\Final Project\\outputs' + modelName
args['datasetPath'] = 'C:\\Users\\saman\\Documents\\Classes\\ECE 243A Brain Computer Interfaces\\Final Project\\competitionData\\decoder_dataset'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

from src.neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)