#0 means left eye and 1 means right eye
train_lr = [1,1,1,0,0,1,1,0,1,0,
            0,1,1,0,1,0,1,1,0,1,
            0,1,1,0,0,1,1,1,0,0,
            0,0,1,1,0,1,1,1,1,0,
            0,0,0,1,0,0,0,0,1,1,
            1,1,1,1,0,1,1,1,1,1,
            0,0,0,1,1,1,1,1,1,1,
            0,0,1,1,0,0,1,0,1,1,
            0,0,1,1,0,1,0,1,0,0,
            1,0,0,0,0,0,0,0,0,0,
            ]

test_lr  = [1,0,0,0,0,0,1,0,0,1,
            0,0,1,0,0,1,1,1,0,0,
            1,1,1,1,0,1,1,1,0,0,
            1,1,1,1,1,1,1,0,1,0,
            1,1,0,1,1,0,0,0,0,1,
            0,1,1,1,1,1,1,1,0,1,
            1,0,1,0,0,0,1,1,1,1,
            0,1,0,0,1,0,1,0,1,1,
            1,0,1,1,1,1,1,1,1,0,
            0,0,1,0,1,1,1,0,1,0,
            ]

#0means 1956x1934 and 1 means 2992x2000
train_size = [0,0,1,1,1,0,1,1,1,1,
              1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,1,1,
              1,0,1,1,1,1,1,1,1,1,
              1,1,1,1,1,0,1,1,1,1,
              1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,0,1,1,1,1,1,
              1,1,1,1,1,1,1,1,1,1,
              ]

test_size  = [1,1,1,1,1,1,0,1,1,1,
              1,1,1,1,1,1,0,1,1,1,
              1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,0,1,1,1,
              1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,1,0,
              1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,0,1,1,1,
              1,1,0,1,1,1,1,1,1,1,
              1,1,0,1,1,1,1,1,1,1,
              ]


#get img path,inputs img index and mode
def get_img_path(index,mode='train'):

    assert index >= 0 and index < 100,    'index need >= 0 and < 100'
    assert mode=='train' or mode=='test',   'mode error'

    if mode == 'train':
        if index < 9:
            path = 'data/training/fundus color images/000{}.jpg'.format(index+1)
        elif index < 99:
            path = 'data/training/fundus color images/00{}.jpg'.format(index+1)
        else:
            path ='data/training/fundus color images/0100.jpg'
    elif mode == 'test':
        if index < 9:
            path = 'data/testing/fundus color images/010{}.jpg'.format(index+1)
        elif index < 99:
            path = 'data/testing/fundus color images/01{}.jpg'.format(index+1)
        else:
            path = 'data/testing/fundus color images/0200.jpg'
    return path