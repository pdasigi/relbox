import numpy, theano, sys, re
from relbox import RelBox
import pickle

dirname = "learned_param5M+187k_500k_glove/"
#dirname = ""
dignorm = False
vocab = []
word_dim = 50
for wrdline in open(dirname+"vocab.txt"):
    lnparts = wrdline.strip().split()
    if lnparts[-1] != 'OOV':
        vocab.append(lnparts[0])
vocab.append('UNK')
rels = [x.strip() for x in open(dirname+"rels.txt")]
#learned_vocab = cPickle.load(open("learned_vocab.pkl", "rb"))
learned_box, learned_proj, learned_rev_proj = pickle.load(open(dirname+"box_param_glove_5.pkl", "rb"))
rel_rep = numpy.eye(len(rels), dtype=theano.config.floatX)
vocab_rep = numpy.eye(len(vocab)+1, dtype=theano.config.floatX)

#relbox = RelBox(learned_box, learned_vocab, rel_rep)
relbox = RelBox(len(vocab), len(rels), word_dim)
relbox.B.set_value(learned_box)
relbox.P.set_value(learned_proj)
relbox.P_hat.set_value(learned_rev_proj)
get_r_pred = relbox.get_r_pred()
get_x_pred = relbox.get_x_pred()
get_y_pred = relbox.get_y_pred()
get_score = relbox.get_score()
testfile = open(sys.argv[1])

for line in testfile:
    #line = sys.stdin.readline()
    lnstrp = line.strip()
    #if lnstrp == "":
    #    break
    parts = lnstrp.split()
    tuples = []
    if len(parts) < 3:
        print "0.0"
        continue
    if len(parts) > 4 and len(parts)%3 == 0:
        if dignorm:
            for i in range(0, len(parts), 3):
                tuples.append((re.sub("[0-9]", "D", parts[i]), parts[i+1], re.sub("[0-9]", "D", parts[i+2])))
        else:
            for i in range(0, len(parts), 3):
                tuples.append((parts[i], parts[i+1], parts[i+2]))
    else:
        if dignorm:
            tuples.append((re.sub("[0-9]", "D", parts[0]), parts[1], re.sub("[0-9]", "D", parts[2])))
        else:
            tuples.append((parts[0], parts[1], parts[2]))
    scores = []
    for t_x, t_r, t_y in tuples:
        try:
            x_ind = vocab.index(t_x.lower())
        except:
            x_ind = len(vocab) - 1
        try:
            y_ind = vocab.index(t_y.lower())
        except:
            y_ind = len(vocab) - 1
        try:
            r_ind = rels.index(t_r.lower())
        except ValueError, e:
            #print >>sys.stderr, "One of %s %s and %s is not in index"%(x, y, r), e
            continue
        scores.append(get_score(x_ind, y_ind, r_ind))
        """r_pred = get_r_pred(x_ind, y_ind)
        x_pred = get_x_pred(y_ind, r_ind)
        y_pred = get_y_pred(x_ind, r_ind)
        rel_probs = sorted(zip(r_pred[0], rels), reverse=True)
        print "r\t%s\t%s\t%s\t%s\t%f"%(t_x, t_y, t_r, rel_probs[0][1], rel_probs[0][0])
        x_probs = sorted(zip(x_pred[0], vocab), reverse=True)
        corr_x_ind = len(vocab) - 1
        for ind, (_, w) in enumerate(x_probs):
            if t_x == w:
                corr_x_ind = ind + 1
        print "x\t%s\t%s\t%s\t%d\t%s"%(t_r, t_y, t_x, corr_x_ind, "\t".join(["%f\t%s"%(p, x) for (p, x) in x_probs[:5]]))
        y_probs = sorted(zip(y_pred[0], vocab), reverse=True)
        corr_y_ind = len(vocab) - 1
        for ind, (_, w) in enumerate(y_probs):
            if t_y == w:
                corr_y_ind = ind + 1
        print "y\t%s\t%s\t%s\t%d\t%s"%(t_x, t_r, t_y, corr_y_ind, "\t".join(["%f\t%s"%(p, y) for (p, y) in y_probs[:5]]))"""
    #print " ".join(["%.5f %s %s %s"%(score, t_x, t_r, t_y) for score, (t_x, t_r, t_y) in zip(scores, tuples)])
    print " ".join(["%.5f"%score for score in scores])
