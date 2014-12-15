import sys, numpy, theano, random
from relbox import RelBox
import cPickle

numpy_rng = numpy.random.RandomState(89677)

def get_vocab_rep(vocab, words_file, embed_file, dim=50):
    init_map = {word: numpy.asarray([float(y) for y in embed.strip().split()]) for word, embed in zip(open(words_file), open(embed_file))}
    vocab_rep = []
    rng_low = -4 * numpy.sqrt(6. / dim)
    rng_high = 4 * numpy.sqrt(6. / dim)
    for word in vocab:
        if word in init_map:
            vocab_rep.append(init_map[word])
        else:
            vocab_rep.append(numpy.asarray(numpy_rng.uniform(low=rng_low, high=rng_high, size=dim)))
    return numpy.asarray(vocab_rep, dtype=theano.config.floatX)

if __name__ == "__main__":
    dense_vocab_rep = False
    dense_rel_rep = False
    word_dim = 50
    #rel_dim = 2
    train_prop = 0.8
    num_iter = 100
    data_file = "data/tuples_train_dev.txt"
    if dense_vocab_rep:
        words_file = "words.lst"
        embed_file = "embeddings.txt"
    infile = open(data_file)
    all_tuples = []
    vocab = set([])
    rels = set([])
    train_tuple_inds = []
    valid_tuple_inds = []
    for line in infile:
        sub, rel, obj, cnt_st = line.strip().split('\t')
        cnt = int(cnt_st)
        all_tuples.extend([(sub, rel, obj)]*cnt)
        vocab.add(sub)
        vocab.add(obj)
        rels.add(rel)
    vocab = list(vocab)
    rels = list(rels)
    vocab_size = len(vocab)
    rel_size = len(rels)
    #init_box_rep = numpy.random.random((word_dim, word_dim, rel_dim))
    # Key idea: Since rel representation is one-hot, rel_dim = rel_size
    #init_box_rep = numpy.random.random((word_dim, word_dim, rel_size))
    #init_vocab_rep = numpy.random.random((vocab_size, dim))
    if dense_vocab_rep:
        init_vocab_rep = get_vocab_rep(vocab, words_file, embed_file, word_dim)
        box_rng_low = -4 * numpy.sqrt(6. / (word_dim + word_dim + rel_size))
        box_rng_high = 4 * numpy.sqrt(6. / (word_dim + word_dim + rel_size))
        init_box_rep = numpy.asarray(numpy_rng.uniform(low=box_rng_low, high=box_rng_high, size=(word_dim, word_dim, rel_size)), dtype=theano.config.floatX)
    else:
        # vocab_size '+ 1' for unk
        sparse_word_dim = vocab_size + 1
        init_vocab_rep = numpy.eye(sparse_word_dim, dtype=theano.config.floatX)
        box_rng_low = -4 * numpy.sqrt(6. / (sparse_word_dim + sparse_word_dim + rel_size))
        box_rng_high = 4 * numpy.sqrt(6. / (sparse_word_dim + sparse_word_dim + rel_size))
        init_box_rep = numpy.asarray(numpy_rng.uniform(low=box_rng_low, high=box_rng_high, size=(sparse_word_dim, sparse_word_dim, rel_size)), dtype=theano.config.floatX)
    print >>sys.stderr, "Initialized Relation Box"
    print >>sys.stderr, "Created initial vocab representation"
    #init_rel_rep = numpy.random.random((rel_size, rel_dim))
    init_rel_rep = numpy.eye(rel_size, dtype=theano.config.floatX)
    print >>sys.stderr, "Created initial relation representation"
    random.shuffle(all_tuples)
    data_size = len(all_tuples)
    train_size = int(data_size * train_prop)
    valid_size = data_size - train_size
    for i, (sub, rel, obj) in enumerate(all_tuples):
        tup_rep = (vocab.index(sub), rels.index(rel), vocab.index(obj))
        if i < train_size:
            train_tuple_inds.append(tup_rep)
        else:
            valid_tuple_inds.append(tup_rep)
    #relbox = RelBox(init_box_rep, init_vocab_rep, init_rel_rep, dense_vocab_rep=dense_vocab_rep, dense_rel_rep=dense_rel_rep)
    #x_update = relbox.update_for_x()
    #y_update = relbox.update_for_y()
    #r_update = relbox.update_for_r()
    relbox = RelBox(init_box_rep, dense_vocab_rep=False, dense_rel_rep=False, train_nce=True)
    nce_train_fun = relbox.train_with_nce()
    get_costs = relbox.get_costs()
    for i in range(num_iter):
        """x_train_costs = []
        y_train_costs = []
        r_train_costs = []
        for s_ind, r_ind, o_ind in train_tuple_inds:
            x_loss = x_update(s_ind, o_ind, r_ind)
            x_train_costs.append(x_loss)
        for s_ind, r_ind, o_ind in train_tuple_inds:
            y_loss = y_update(s_ind, o_ind, r_ind)
            y_train_costs.append(y_loss)
        for s_ind, r_ind, o_ind in train_tuple_inds:
            r_loss = r_update(s_ind, o_ind, r_ind)
            r_train_costs.append(r_loss)
        avg_train_cost = (sum(x_train_costs)/train_size, sum(y_train_costs)/train_size, sum(r_train_costs)/train_size)"""
        nce_losses = []
        for s_ind, r_ind, o_ind in train_tuple_inds:
            nce_loss = nce_train_fun(s_ind, o_ind, r_ind)
            nce_losses.append(nce_loss)
        avg_train_cost = sum(nce_losses)/train_size
        #avg_train_cost = ((sum([x[0] for x in train_costs])+0.0)/train_size, (sum([x[1] for x in train_costs])+0.0)/train_size, (sum([x[2] for x in train_costs])+0.0)/train_size)
        #print >>sys.stderr, "Finished epoch %d, train costs are %f, %f, %f"%(i+1, avg_train_cost[0], avg_train_cost[1], avg_train_cost[2])
        print >>sys.stderr, "Finished epoch %d, train cost is %f"%(i+1, avg_train_cost)
        if (i+1)%10 == 0:
            valid_costs = []
            for s_ind, r_ind, o_ind in valid_tuple_inds:
                x_loss, y_loss, r_loss = get_costs(s_ind, o_ind, r_ind)
                valid_costs.append((x_loss, y_loss, r_loss))

            avg_valid_cost = ((sum([x[0] for x in valid_costs])+0.0)/valid_size, (sum([x[1] for x in valid_costs])+0.0)/valid_size, (sum([x[2] for x in valid_costs])+0.0)/valid_size)
            print >>sys.stderr, "\tvalid costs are %f, %f, %f"%(avg_valid_cost[0], avg_valid_cost[1], avg_valid_cost[2])
    words_file = open("vocab.txt", "w")
    for word in vocab:
        print >>words_file, word
    words_file.close()
    rels_file = open("rels.txt", "w")
    for rel in rels:
        print >>rels_file, rel
    rels_file.close()
    learned_box = relbox.B.get_value()
    learned_vocab = relbox.vocab.get_value()
    learned_rel = relbox.rel.get_value()
    box_file = open("nce_box.pkl", "wb")
    cPickle.dump(learned_box, box_file)
    #vocab_file = open("learned_vocab.pkl", "wb")
    #cPickle.dump(learned_vocab, vocab_file)
    #rel_file = open("learned_rel.pkl", "wb")
    #cPickle.dump(learned_rel, rel_file)
    print >>sys.stderr, "Done!"
