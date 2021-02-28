# Color Usage Analysis in Paintings

Machine learning based analysis of color usage by famous artists (like Jackson Pollock) in their paintings

This repo builds upon [Andrea Lalenti's work](https://towardsdatascience.com/clustering-pollock-1ec24c9cf447), also available [here](https://gitlab.com/andrea.ialenti/pollock-analysis/-/tree/master/)

## Instructions to setup project and python environment:

Clone this repository using the following command:

```console
git clone https://github.com/smedury/color-usage-analysis-in-paintings.git
```

Switch working directory to project and install pip requirements using the following command:

```console
cd color-usage-analysis-in-paintings
pip3 install -r requirements.txt
```

## Run instructions:

This project can analyse paintings of artists:
- Jackson Pollock
- Frida Kahlo 

using kmeans clustering analysis.

There are 2 main scripts, one for each artist.

Run the following command to:
- gather images using scraper module 
- analyze color usage using clustering (K-Means or DBSCAN)
- gather colors proportions per year analysis:

```console
python3 main_<aritst_last_name>_<clustering_type>.py
```

For instance, to run *kmeans* cluster analysis for artist *Frida Kahlo's* painting:

```console
python3 main_kahlo_kmeans.py
```
And to run *kmeans* cluster analysis for artist *Jackson Pollock's* paintings:
```console
python3 main_pollock_kmeans.py
```

### Output

Output images can be found in folder named *data*, created in the project's root directory.

Top 20 colors used by the artist are identified through clustering. And their proportions are determined per painting and per year.

## Potential Issues and fixes

- When installing on Windows 10 using Git bash terminal (with anaconda installed), you may encounter the following error:

    ```console
    building 'twisted.test.raiser' extension error: 
    Microsoft Visual C++ 14.0 is required. 
    Get it with "Build Tools for Visual Studio": https://visualstudio.microsoft.com/downloads/
    ---------------------------------------- 
    ERROR: Failed building wheel for Twisted
    ```
    Install MS Visual Studio using available for free at this [link](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16) 