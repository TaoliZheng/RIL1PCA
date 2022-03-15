% Vaishali R | VIT University   %
% Datasets for Feature Selection%
% Email: vrv.vaishali@gmail.com%
%
%%

clc; clear all;


% 1. Zoo Dataset
zoo=load('data\zoo.dat');
fprintf('Zoo.dat Loaded Successfully \n');

%2. Wine Dataset
winetemp=load('data\wine.mat');
wine=winetemp.A;
fprintf('wine.mat Loaded Successfully \n');

% 3. Votes Dataset

votestemp=load('data\votes.mat');
votes=votestemp.votes;
votes=knnimpute(votes); % KNN impute handles missing data in the votes dataset
fprintf('votes.mat Loaded Successfully \n');

% 4. SPECT dataset

specttemp=load('data\spect.mat');
spect=specttemp.output;
fprintf('spect.mat Loaded Successfully \n');

% 5. Semeion Dataset
semeion=load('data\semeion.dat');
fprintf('semeion.dat Loaded Successfully \n');

% 6. ILPD dataset
ilpdtemp=load('data\lipid.mat');
ilpd=ilpdtemp.ilpd;
fprintf('ilpd Loaded Successfully \n');

% 7. isolet5 Dataset

isolet=load('data\isolet5.dat');

fprintf('isolet5.dat Loaded Successfully \n');


% 8. Ionosphere Dataset

ionosphere=load('data\ionosphere.csv');
fprintf('ionosphere.csv Loaded Successfully \n');

% 9. Heart Disease dataset
heart=load('data\heart.dat');
fprintf('heart.dat Loaded Successfully \n');

% 10. Glass Dataset
glass= load('data\glass.dat');
fprintf('glass.dat Loaded Successfully \n');

% 11. COIL20 Dataset
coiltemp=load('data\coil.mat');
coil=coiltemp.b;
fprintf('coil.mat Loaded Successfully \n');

% 12. Clean1 Dataset (MUSK)
clean1= load('data\clean1.csv');
fprintf('clean1.csv Loaded Successfully \n');


% 13. BreastEW Dataset
btemp=load('data\breastEW.mat');
breastEW=btemp.data;
fprintf('breastEW.mat Loaded Successfully \n');


fprintf('All datasets are loaded Successfully!\n');
