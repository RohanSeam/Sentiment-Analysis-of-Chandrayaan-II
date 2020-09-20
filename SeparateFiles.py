

lines_per_file = 1
smallfile = None
with open('RohanProject/pos/cv9_posTest.txt', encoding="utf8") as bigfile:
           # Splitted_txt_sentoken/cv9_compound_neg_27K.txt
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = 'RohanProject/pos/cv9_{}.txt'.format(lineno + lines_per_file)
            smallfile = open(small_filename, "w", encoding="utf8")
        smallfile.write(line)
    if smallfile:
        smallfile.close()