# kerastfML

# step 1
# Preparing directories and training data set before first run

Before the kwasir-dataset archive is extracted, the directory tree will look like this

.
├── data
│   ├── .....
├── learn
│   ├── .....
├── model
│   ├── .....
├── main.py
└── README.md

Extract kwasir-dataset in to the same folder where the "main.py" file exists
After doing it the directory tree shold look like the following

.
├── data
│   ├── .....
├── kvasir-dataset
│   ├── dyed-lifted-polyps
│   ├── dyed-resection-margins
│   ├── esophagitis
│   ├── normal-cecum
│   ├── normal-pylorus
│   ├── normal-z-line
│   ├── polyps
│   └── ulcerative-colitis
├── learn
│   ├── .....
├── model
│   ├── .....
├── main.py
└── README.md

If the data set is not placesd within the right directory with right directory structure, it will be caused for runtime errors
