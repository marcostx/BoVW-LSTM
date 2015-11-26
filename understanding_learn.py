from os.path import exists, isdir, basename, join, splitext
import sift
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
import libsvm
from cPickle import dump, HIGHEST_PROTOCOL
import argparse


EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
DATASETPATH = '../dataset'
PRE_ALLOCATION_BUFFER = 1000  # for sift
HISTOGRAMS_FILE = 'trainingdata.svm'
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODEBOOK_FILE = 'codebook.file'

def get_categories(datasetpath):
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                  if isdir(files)]

    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]

    return cats

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='path to the dataset', required=False, default=DATASETPATH)
    args = parser.parse_args()
    return args

def get_imgfiles(path):
    all_files = []

    all_files.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files

def extractSift(input_files):
    print "extracting Sift features"
    all_features_dict = {}

    for i, fname in enumerate(input_files):
        features_fname = fname + '.sift'

        if exists(features_fname) == False:
            print "calculating sift features for", fname
            sift.process_image(fname, features_fname)

        print "gathering sift features for", fname,
        locs, descriptors = sift.read_features_from_file(features_fname)
        print descriptors.shape
        all_features_dict[fname] = descriptors

    return all_features_dict

def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 128))
    return array

def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words

def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print 'nclusters have been reduced to ' + str(nwords)
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)

if __name__ == '__main__':
	
	print "## analisando parametros"
	args = parse_arguments()
	datasetpath = args.d

	categories = get_categories(datasetpath)

	size_cat = len(categories)
	print "procurando pastas em " + datasetpath

	if size_cat < 1:
		print "Error!"

	all_files = []
	all_files_labels = {}
	all_features = {}
	cat_label = {}
	for cat, label in zip(categories,range(size_cat)):
		# datapath/class-i
		cat_path = join(datasetpath, cat)
		# cat_files recebe as imagens da pasta correspondente a classe
        cat_files = get_imgfiles(cat_path)
        # cat_features recebe as features das imagens coletadas
        cat_features = extractSift(cat_files)
        # all_files contem todos os arquivos de todas as pastas
        all_files = all_files + cat_files
        # all_features contem todas as features de todas as imagens
        all_features.update(cat_features)
        # car_label recebe o rotulo de cada classe 
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label
	
	print "## Calcular as palavras visuais via k-means"
	# Transforma dicionario de features em um vetor numpy
	all_features_array = dict2numpy(all_features)
	# Numero de features
	nfeatures = all_features_array.shape[0]
	# Numero de clusters
	nclusters = int(sqrt(nfeatures))

	# Executando kmeans e obtendo o codebook
	codebook, distortion = vq.kmeans(all_features_array,nclusters,thresh=K_THRESH)		

	# Escrevendo o codebook
	with open(datasetpath + CODEBOOK_FILE, 'wb') as f:
		dump(codebook, f, protocol=HIGHEST_PROTOCOL)

	print "## Calcular os histogramas de palavras visuais para cada imagem"
	all_word_histgrams = {}
	for imagefname in all_features:
		word_histgram = computeHistograms(codebook, all_features[imagefname])
		all_word_histgrams[imagefname] = word_histgram

	print "## Escrever os histogramas no arquivo para passar para o SVM"
	writeHistogramsToFile(nclusters,
						  all_files_labels,
						  all_files,
                          all_word_histgrams,
                          datasetpath + HISTOGRAMS_FILE)

	print "Treinar SVM"
	c, g, rate, model_file = libsvm.grid(datasetpath + HISTOGRAMS_FILE,
                                         png_filename='grid_res_img_file.png')

	print "--------------------"
	print "Resultados: "
	print "Arquivo: " + datasetpath + model_file
	print "Codebook: " + datasetpath + CODEBOOK_FILE
	print "Categoria      ==>  Rotulo"
	for cat in cat_label:
		print '{0:13} ==> {1:6d}'.format(cat, cat_label[cat])
