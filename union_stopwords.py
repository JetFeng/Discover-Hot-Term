# coding='utf-8'

stop_list1 = []
stop_list2 = []
with open('stop_words1.txt', 'r', encoding='utf8') as fp:
    stop_list1 = list(fp.readlines())

with open('stop_words2.txt', 'r', encoding='utf8') as fp:
    stop_list2 = list(fp.readlines())

stop_list2.reverse()
for w in stop_list2:
    if w not in stop_list1:
        stop_list1.append(w)

with open('stop_list.txt', 'w', encoding='utf8') as fw:
    fw.writelines(stop_list1)