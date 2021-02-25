# Color Usage Analysis in Paintings

Machine learning based analysis of color usage by famous artists (like Jackson Pollock) in their paintings

This repo builds upon [Andrea Lalenti's work](https://towardsdatascience.com/clustering-pollock-1ec24c9cf447), also available [here](https://gitlab.com/andrea.ialenti/pollock-analysis/-/tree/master/)

## Instructions to build and run this project:

Clone this repository using the following command:

```console
git clone https://github.com/smedury/color-usage-analysis-in-paintings.git
```

Switch working directory to project and install pip requirements using the following command:

```console
cd color-usage-analysis-in-paintings
pip3 install -r requirements.txt
```

Run the following command to:
- gather images using scraper module 
- analyze color usage using K-Means clustering
- gather colors proportions per year analysis:

```console
python3 main.py
```