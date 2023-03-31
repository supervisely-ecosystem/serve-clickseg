from bs4 import BeautifulSoup

with open("src/models.html") as f:
    text = f.read()
soup = BeautifulSoup(text)

modes = ["CDNet", "FocalClick", "Baseline"]
comb_dataset = "COCO, LVIS, ADE20K, MSRA10K, DUT, YoutubeVOS, ThinObject, HFlicker"
comb = {"CombinedDatasets": comb_dataset, "Combined+Dataset": comb_dataset}
model_zoo = []

for table, mode in zip(soup.find_all("table"), modes):
    for row in table.find_all("tbody"):
        row = row.find("tr")
        links = row.find_all("a")
        assert len(links) == 1
        url = links[0].attrs["href"]
        cells = row.find_all("td")
        dataset = cells[0].text
        if dataset in comb:
            dataset = comb[dataset]
        model_name = cells[1].text
        print(cells[1].text)
        model_name, size = model_name.split("(")
        size = size[:-1]
        model_id = model_name.lower() + "_" + str(len(model_zoo))
        info = {
            "model_id": model_id,
            "model": model_name,
            "mode": mode,
            "weights_url": "",
            "trained on": dataset,
            "size": size,
        }
        model_zoo.append(info)

import json

with open("src/models.json", "w") as f:
    json.dump(model_zoo, f, indent=4)
