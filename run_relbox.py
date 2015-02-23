import sys, random, gzip
from relbox import RelBox
import cPickle
import operator
import numpy

if __name__ == "__main__":
    word_dim = 50
    oov_cond = "prop" # prop or freq
    oov_prop = 0.1 # oov_prop of the words with low frequencies will be considered oov
    oov_freq_limit = 2
    train_prop = 0.8
    num_iter = 100
    init_P = "rand" # rand or embed
    numpy_rng = numpy.random.RandomState(89677)
    data_file = sys.argv[1]
    infile = open(data_file)
    all_tuples = []
    vocab_freq_map = {}
    oov_words = [] 
    rels = set([])
    train_tuple_inds = []
    valid_tuple_inds = []
    for line in infile:
        sub, rel, obj, cnt_st = line.strip().split('\t')
        cnt = int(cnt_st)
        all_tuples.extend([(sub, rel, obj)]*cnt)
        #all_tuples.extend([(sub, rel, obj)])
        if sub in vocab_freq_map:
            vocab_freq_map[sub] += cnt
        else:
            vocab_freq_map[sub] = cnt
        if obj in vocab_freq_map:
            vocab_freq_map[obj] += cnt
        else:
            vocab_freq_map[obj] = cnt
        rels.add(rel)
    vocab = {} # vocab is a map : word -> index
    sorted_vocab = sorted(vocab_freq_map.items(), key=operator.itemgetter(1))
    init_vocab_size = len(vocab_freq_map)
    if oov_cond == "prop":
        oov_num = int(len(sorted_vocab) * oov_prop)
        vocab_index = 0
        for ind, (word, freq) in enumerate(sorted_vocab):
            if ind > oov_num:
                #vocab.append(word)
                vocab[word] = vocab_index
                vocab_index += 1
            else:
                oov_words.append(word)
        print >>sys.stderr, "Considering the bottom %f as OOV"%(oov_prop)
    else:
        vocab_index = 0
        for word, freq in sorted_vocab:
            if freq >= oov_freq_limit:
                #vocab.append(word)
                vocab[word] = vocab_index
                vocab_index += 1
            else:
                oov_words.append(word)
        print >>sys.stderr, "Considering words with frequency less than %d as OOV"%(freq)
    vocab_size = len(vocab)
    #vocab.append('UNK') # Last word in UNK
    vocab['UNK'] = vocab_size # Last word in UNK
    vocab_size += 1
    print >>sys.stderr, "Vocab size changed from %d to %d"%(init_vocab_size, vocab_size)
    rels = list(rels)
    words_file = open("vocab.txt", "w")
    for word, freq in sorted_vocab:
        if word in oov_words:
            print >>words_file, word, freq, "OOV"
        else:
            print >>words_file, word, freq
    words_file.close()
    rels_file = open("rels.txt", "w")
    for rel in rels:
        print >>rels_file, rel
    rels_file.close()
    rel_size = len(rels)
    random.shuffle(all_tuples)
    data_size = len(all_tuples)
    train_size = int(data_size * train_prop)
    valid_size = data_size - train_size
    for i, (sub, rel, obj) in enumerate(all_tuples):
        if sub in vocab:
            #sub_ind = vocab.index(sub)
            sub_ind = vocab[sub]
        else:
            sub_ind = vocab['UNK']
        if obj in vocab:
            #obj_ind = vocab.index(obj)
            obj_ind = vocab[obj]
        else:
            #obj_ind = len(vocab) - 1
            obj_ind = vocab['UNK']
        tup_rep = (sub_ind, rels.index(rel), obj_ind)
        if i < train_size:
            train_tuple_inds.append(tup_rep)
        else:
            valid_tuple_inds.append(tup_rep)
    if init_P == "embed":
        print >>sys.stderr, "Initializing projection with embedding at %s"%(sys.argv[2])
        randrep = numpy.asarray(numpy_rng.uniform(low = -4 * numpy.sqrt(6. / word_dim), high = 4 * numpy.sqrt(6. / word_dim), size=word_dim))
        vocab_rep = [randrep] * vocab_size
        embed_file = gzip.open(sys.argv[2])
        in_vocab = set()
        for line in embed_file:
            lineparts = line.strip().split()
            wrd = lineparts[0]
            if wrd in vocab:
                vocab_rep[vocab[wrd]] = numpy.asarray([float(x) for x in lineparts[1:]])
                in_vocab.add(wrd)
        print >>sys.stderr, "Embedding has %f coverage"%(float(len(in_vocab))/vocab_size)
        relbox = RelBox(vocab_size, rel_size, word_dim, num_noise_samples=1, init_dense_vocab=numpy.asarray(vocab_rep))
        pkl_ext = sys.argv[2].split('/')[-1].split('.')[0]
    else:
        relbox = RelBox(vocab_size, rel_size, word_dim, num_noise_samples=1)
        pkl_ext = "rand"
        print >>sys.stderr, "Initializing projection randomly"
    nce_train_fun = relbox.train_with_nce_margin()
    get_costs = relbox.get_costs()
    for i in range(num_iter):
        nce_losses = []
        for s_ind, r_ind, o_ind in train_tuple_inds:
            nce_loss = nce_train_fun(s_ind, o_ind, r_ind)
            nce_losses.append(nce_loss)
        avg_train_cost = sum(nce_losses)/train_size
        print >>sys.stderr, "Finished epoch %d, train cost is %f"%(i+1, avg_train_cost)
        if avg_train_cost == 0:
            break
        if (i+1)%5 == 0:
            learned_param = (relbox.B.get_value(), relbox.P.get_value(), relbox.P_hat.get_value())
            box_file = open("box_param_%s_%d.pkl"%(pkl_ext, i+1), "wb")
            cPickle.dump(learned_param, box_file)
            print >>sys.stderr, "Dumped relbox"
            box_file.close()
        if (i+1)%10 == 0:
            valid_costs = []
            for s_ind, r_ind, o_ind in valid_tuple_inds:
                x_loss, y_loss, r_loss = get_costs(s_ind, o_ind, r_ind)
                valid_costs.append((x_loss, y_loss, r_loss))

            avg_valid_cost = ((sum([x[0] for x in valid_costs])+0.0)/valid_size, (sum([x[1] for x in valid_costs])+0.0)/valid_size, (sum([x[2] for x in valid_costs])+0.0)/valid_size)
            print >>sys.stderr, "\tvalid costs are %f, %f, %f"%(avg_valid_cost[0], avg_valid_cost[1], avg_valid_cost[2])
    learned_param = (relbox.B.get_value(), relbox.P.get_value(), relbox.P_hat.get_value())
    box_file = open("box_param_%s.pkl"%(pkl_ext), "wb")
    cPickle.dump(learned_param, box_file)
    print >>sys.stderr, "Done!"
