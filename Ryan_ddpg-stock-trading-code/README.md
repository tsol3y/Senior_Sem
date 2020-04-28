# Both DDPG and Meta-Critic are running!

## Erik's Notes

I have gotten the code to run and have saved all of the dependencies in ddpg.yaml. You can create a new conda environment that copies the one that I used to get this code running by:

```
conda env create -f ddpg_env.yaml
```

This will install all of the correct versions of the packages required to run this code.

You will have to install torch separately however. I used torch 1.5.0, and you can install that using:
```
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

You may run into an issue installing tables. For tables to work on windows, you need Visual Studio C++ build tools. If you do not already have that installed, you can install that from here: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16

Lastly, to register the environment as a jupyter notebook kernel so it shows up you run:

```
$ conda activate cenv
(cenv)$ conda install ipykernel
(cenv)$ ipython kernel install --user --name=<any_name_for_kernel>
(cenv)$ conda deactivate
```

## DDPG is running!
These are a couple package versions that are necessary for DDPG to run. Checkout freeze.txt to see my packages and their versions. Those are just all of my pip packages, so many are irrelevant, but use it as a reference. Just some notable packages that I remember that I've had to download or change versions on are:
1. gym (make sure you have my version of Gym in the freeze)
2. seaborn (make sure you have my version of seaborn in the freeze)
3. cvxopt
4. h5py
5. pandas-datareader
6. scikit-image
7. statsmodels
8. tables

```
pip install gym==0.9.3 seaborn==0.8 cvxopt==1.2.5 h5py==2.9.0 pandas==0.25.1 pandas-datareader==0.8.1 scikit-image==0.16.2 statsmodels==0.10.1 tables==3.6.1
```

Also, as you run each cell in jupyter notebook, just install the modules it says you are missing using pip install. For one of them you may need to use:
```
$>pip install --user packageNameHere
```
since a package will try to download onto protected parts of your hardrive. --user tells pip to install the package under just your user account.

To give an idea of what is in the ddpg-stock-trading-code folder now to get the code to work, I've added the DeepRL and UniversalPortfolios repositories since there are many dependencies this ddpg repository requires in those. I've also edited some of the import paths and even code in the repositories to get it all to work.

**A really important note to save some time!!**  
Instructions on running a successful test for pytorch-deeprl-DDPG-EIIE.ipynb:
1. Run through each cell in jupyter downloading the necessary modules and changing versions according to my versions or the requirements/repo-freeze files in the requirements folder.
2. Once you get to the cell labeled Train, the current setup has a saving scheme that will take forever for the rest of the code to work. We want to change the saving schema to be more often so we can continue on to make sure all of the code works. So, use the quicktest lines rather than the originals.

![quicktestcode](readme_files/quicktest.PNG)

3. Next, allow the training to go for maybe 13 episodes and then hit the interupt kernal button. Interruption doesn't ruin the code since the training files are saved tracking results.
4. Then you can go on past the training cell and get those cells working.

**If you have any questions let me know (Ryan)!! Maybe I can help.**


## Meta-critic is running!
Run the mvc using Python 3.7, Pytorch 1.3.1 and without cuda with...
```
root>python meta-critic-cartpole-code/mvn_cartpole.py
```

<br/><br/><br/><br/><br/><br/><br/>

Template

# Meta-Critic For Stock Trading

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
