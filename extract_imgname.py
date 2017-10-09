import os

def extract_url(oldfile, newfile):
    with open(oldfile) as f:

        # read file
        urls = file.readlines(f)
        # create a variable to store extracted urls
        extract_urls = []
        # extract urls
        for i in range(len(urls)):
            url = urls[i]
            for j in range(len(url)):
                if url[j] == '/':
                    if url[(j + 1) : (j + 10)] == 'sf_orgres':
                        start = j + 10
                        break
            for j in range(len(url))[::-1]:
                if url[j] == 'g':
                    end = j + 1
                    break
            extract_urls.append(url[start:end] + '\n')

    # if has extract file, remove it
    if os.path.isdir(newfile):
        os.remove(newfile)

    # write extracted urls to new file
    with open(newfile, 'w') as f:
        f.writelines(extract_urls)
        print 'ok'

extract_url('data/all_sanfran_netvlad_trn_fr.txt', 'data/extract_all_sanfran_netvlad_trn_fr.txt')
extract_url('data/val_sanfran_netvlad_trn_fr.txt', 'data/extract_val_sanfran_netvlad_trn_fr.txt')