def dataset_init(preprocessing_args):
    if preprocessing_args.dataset == "cds_and_vinyl" or "kindle":
        from scripts.preprocessing.amazon_dataset_preprocessing import amazon_preprocessing
        rec_dataset = amazon_preprocessing(preprocessing_args)
    return rec_dataset
