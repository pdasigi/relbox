import numpy, theano, sys
from relbox import RelBox
import cPickle

vocab = [x.strip() for x in open("vocab.txt")]
rels = [x.strip() for x in open("rels.txt")]
#learned_vocab = cPickle.load(open("learned_vocab.pkl", "rb"))
learned_box = cPickle.load(open("box.pkl", "rb"))
rel_rep = numpy.eye(len(rels), dtype=theano.config.floatX)
vocab_rep = numpy.eye(len(vocab)+1, dtype=theano.config.floatX)

#relbox = RelBox(learned_box, learned_vocab, rel_rep)
relbox = RelBox(learned_box, vocab_rep, rel_rep)
get_r_pred = relbox.get_r_pred()
get_x_pred = relbox.get_x_pred()
get_y_pred = relbox.get_y_pred()
testfile = open("tuples_test.txt")

for line in testfile:
    #line = sys.stdin.readline()
    lnstrp = line.strip()
    if lnstrp == "":
        break
    t_x, t_r, t_y, _ = lnstrp.split()
    try:
        x_ind = vocab.index(t_x.lower())
        y_ind = vocab.index(t_y.lower())
        r_ind = rels.index(t_r.lower())
    except ValueError, e:
        #print >>sys.stderr, "One of %s %s and %s is not in index"%(x, y, r), e
        continue
    r_pred = get_r_pred(x_ind, y_ind)
    x_pred = get_x_pred(y_ind, r_ind)
    y_pred = get_y_pred(x_ind, r_ind)
    rel_probs = sorted(zip(r_pred[0], rels), reverse=True)
    print "r\t%s\t%s\t%s\t%s\t%f"%(t_x, t_y, t_r, rel_probs[0][1], rel_probs[0][0])
    x_probs = sorted(zip(x_pred, vocab+['unk']), reverse=True)
    print "x\t%s\t%s\t%s\t%f\t%s"%(t_r, t_y, t_x, vocab.index(t_x), "\t".join(["%f\t%s"%(p, x) for (p, x) in x_probs[:5]]))
    y_probs = sorted(zip(y_pred, vocab+['unk']), reverse=True)
    print "y\t%s\t%s\t%s\t%f\t%s"%(t_x, t_r, t_y, vocab.index(t_y), "\t".join(["%f\t%s"%(p, y) for (p, y) in y_probs[:5]]))
