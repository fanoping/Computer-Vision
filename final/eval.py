from util import cal_avgerr, readPFM


files = ['./synthe/TL{}_output_noad.pfm'.format(i) for i in range(10)]
gts = ['./data/Synthetic/TLD{}.pfm'.format(i) for i in range(10)]

total = 0
for idx, (f, g) in enumerate(zip(files, gts)):
    gt = readPFM(g)
    result = readPFM(f)

    err = cal_avgerr(gt, result)
    total += err
    print("{}:".format(idx), err)

print("Ave:", total/10)
