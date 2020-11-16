import csv
header = ['name', 'number']
dictionaries = []
with open("F:\\语料\\动词字典2.csv", 'w', newline='', encoding='utf-8') as out_file:
    f_csv = csv.DictWriter(out_file, header)
    f_csv.writeheader()
    with open("F:\\语料\\动词字典.txt", "rb") as in_file:
        text = in_file.readlines()
        for item in text:
            item = item.decode("utf-8")
            if item[0:3] == '---':
                for i in item:
                    if i == '[':
                        break
                    item =item[1:]
                item = item[1:4]
                number = item
                # print(item)
            elif item[0] != "-":
                item_new = item
                for i in item_new:
                    if i == '[':
                        break
                    item_new =item_new[1:]
                item_number = item_new[1:4]
                print(item_number)
                if item_number == "":
                    count = 0
                    for i in item:
                        count = count + 1
                        if i == ' ':
                            break
                    item = item[0:count]
                    item = item.strip()
                    # print(item)
                    dict = {}
                    dict['number'] = number
                    dict['name'] = item
                    dictionaries.append(dict)
                else:
                    count = 0
                    for i in item:
                        count = count + 1
                        if i == ' ':
                            break
                    item = item[0:count]
                    item = item.strip()
                    # print(item)
                    dict = {}
                    dict['number'] = item_number
                    dict['name'] = item
                    dictionaries.append(dict)

                    dict2 = {}
                    dict2['number'] = number
                    dict2['name'] = item
                    dictionaries.append(dict2)
    f_csv.writerows(dictionaries)


