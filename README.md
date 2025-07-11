
# Applying Causal Mechanisms for Improving Out-of-Distribution Generalisation in Process Prediction

Two architectures are supported：LSTM（RNN）and Transformer.


## supported models
- LSTM
- Transformer
- Graph Transformer

## quick start

### 1. install dependencies
```bash
pip install -r requirements.txt
```

### 2. convert data
```bash
# IGNORE!! convert data from csv/xes to json/jsonl
python data_converter.py --input your_data.csv --output your_data.json --format json \
    --feature-cols feature1 feature2 feature3 --label-col label --domain-col domain
```

### 3. run without adain module
```bash
python run_all_wo_adain.py


```

### 4. run with adain module
```bash
python run_all_w_adain.py
```

## project structure

```
domain_generalization_project/
├── domain_generalization/          # core package
│   ├── models/                     # models implementation
│   ├── data_loader.py             
│   ├── trainer.py                 
│   ├── evaluator.py               
│   ├── pipeline.py                
│   └── utils.py                  
├── results/
├── processed_data/
├── main.py                        # run models without adain     
|── main_adain.py                  # run models with adain      
├── data_converter.py   
├── data_preprocess.py             # data preprocessing           
├── requirements.txt               
├── config_example.json            # config example
├── data_format_example.json       # data form example

```
