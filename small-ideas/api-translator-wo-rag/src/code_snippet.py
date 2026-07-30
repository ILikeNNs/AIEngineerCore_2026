pythoncode = """
lda = sagemaker.estimator.Estimator(
    container,
    role,
    output_path="s3://{}/{}/output".format(bucket, prefix),
    train_instance_count=1,
    train_instance_type="ml.c4.2xlarge",
    sagemaker_session=session,
)

lda.set_hyperparameters(
    num_topics=num_topics,
    feature_dim=vocabulary_size,
    mini_batch_size=num_documents_training,
    alpha0=1.0,
)

lda.fit({"train": s3_train_data})
"""