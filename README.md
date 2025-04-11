This project develops a new diffusion model for generating 2D images similar to a training set. It makes use of a statistical property of natural image sets, namely scale invariance of the variance tensor. The model diffuses images using scale invariant noise whose scaling parameter is fine-tuned to match that of the dataset. This allows us to draw from a noise distribution that is closer (in a quantifiable sense) to the dataset, potentially improving inference.

The project is broken into 4 Jupyter Lab notebooks. Start with CD_Scale_Invariance.ipynb and follow the links in there to the other notebooks. The project is meant to be viewable without running all the cells. If you do want to run the cells, note that the training runs took 2-3 hours each on a Macbook M1 Pro with 16GB memory. 

Feedback is appreciated. Contact me at arandono[at]gmail
