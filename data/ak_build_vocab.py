import nltk
import argparse
from collections import Counter
import pickle
import json


def build_vocab(args):
    with open(args.dataset, 'r') as f:
        dataset_coco = json.load(f)

    imgs = dataset_coco['images']

    count_thr = args.count_thr
    # count up the number of words
    counts = {}
    for img in imgs:
            for sent in img['sentences']:
                for w in sent['tokens']:
                    counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for n, w in cw if n <= count_thr]
    vocab = [w for n, w in cw if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<unk>')
    vocab.append('<start>')
    vocab.append('<end>')

    # for img in imgs:
    #     img['final_captions'] = []
    #     for sent in img['sentences']:
    #         txt = sent['tokens']
    #         caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
    #         img['final_captions'].append(caption)
    print 'Vocabulary size is %d\n'%(len(vocab))
    return vocab


def token_id(vocab, args):
    token2id={}
    id2token={}
    for i, w in enumerate(vocab):
        token2id[w] = i
        id2token[i] = w

    with open(args.vocab_file, 'w') as f:
        pickle.dump({'token2id': token2id, 'id2token': id2token}, f)


def build_captions(args):

    with open(args.dataset, 'r') as f:
        dataset_coco = json.load(f)
    with open(args.vocab_file, 'r') as f:
        vocab = pickle.load(f)

    def get_cap_id(images_list):
        final_train_caption_ids = []
        for image in images_list:
            for sentence in image['sentences']:
                cap_id = []
                tokens = sentence['tokens']
                if len(tokens) > 18:
                    tokens = tokens[0:18]
                tokens = ['<start>'] + tokens + ['<end>']
                for token in tokens:
                    if token in vocab['token2id'].keys():
                        cap_id.append(vocab['token2id'][token])
                    else:
                        cap_id.append(vocab['token2id']['<unk>'])
                image_cap_dict = {'image_id': image['cocoid'], 'filename': image['filename'], 'caption_id': cap_id}
                final_train_caption_ids.append(image_cap_dict)
        return final_train_caption_ids

    train_images = []
    val_images = []
    test_images = []
    images = dataset_coco['images']
    for image in images:
        if image['split'] == 'test':
            test_images.append(image)
        elif image['split'] == 'val':
            val_images.append(image)
        else:
            train_images.append(image)
    if args.is_serve:
        final_train_images = train_images.extend(val_images.extend(test_images[0:3000]))
        final_val_images = test_images[3000:]
        final_train_caption_ids = get_cap_id(final_train_images)
        final_val_caption_ids = get_cap_id(final_val_images)
        with open(args.caption_id_file, 'w') as f:
            pickle.dump({'train':final_train_caption_ids, 'val': final_val_caption_ids}, f)
    else:
        final_train_images = train_images
        final_val_images = val_images
        final_test_images = test_images
        final_train_caption_ids = get_cap_id(final_train_images)
        final_val_caption_ids = get_cap_id(final_val_images)
        final_test_caption_ids = get_cap_id(final_test_images)
        with open(args.caption_id_file, 'w') as f:
            pickle.dump({'train':final_train_caption_ids, 'val': final_val_caption_ids,
                         'test': final_test_caption_ids}, f)


def main(args):
    vocab = build_vocab(args)
    token_id(vocab, args)
    build_captions(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='./files/dataset_coco.json')
    parser.add_argument('--vocab_file', default='./files/vocab.pkl')
    parser.add_argument('--is_serve', default=False)
    parser.add_argument('--caption_id_file', default='./files/caption_id.pkl')
    parser.add_argument('--count_thr', default=5)

    args = parser.parse_args()
    main(args)