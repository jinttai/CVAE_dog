clc; clear;

cvae_training_data = readtable("v1.csv");
cvae_epoch = cvae_training_data.epoch;
cvae_loss = cvae_training_data.train_loss;


figure(1)
plot(cvae_epoch, cvae_loss);
title("CVAE Training Result")
xlabel("Training epoch");
ylabel("Loss");