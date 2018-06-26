clear;

%%calculate training set for svm, should be a column vector in 256 dimension
%%basically apply all f (or g) to ramdom RV
%load('K.mat', 'K');
K=4;
load('K_score_layer1-24_iteration20.mat', 'f');
load('M6_PreProc.mat','M6_RImg', 'M6_Labels');
load('model.mat', 'SVModel')
f = reshape(f,[8,8]);


M6L1_SVec=zeros(60000, 64*K);
for Img = 1:60000
    for row=1:8
        for col=1:8
            for layer=1:K
                %% get the function for the chosen RV
                func = f{row,col}(:,layer);
                
                %% M6L1_SVec is our training set expressed in the 256 dimension vector from
                %%% for each img, using function matrics to compute 256
                %%% value
                M6L1_SVec(Img, (row-1)*32+(col-1)*K+layer) = func(M6_RImg(Img,row,col));
            end
        end
    end
end
% 
% 
% disp('----start training--------')
% %% SVM fit model
% SVModel=fitcecoc(M6L1_SVec, M6_Labels);
% disp('----finish training--------')

%% predict
[g,~]=predict(SVModel, M6L1_SVec);
disp('-------error in prediction----------')
sum((g~= M6_Labels))
disp('-------percentage error in prediction in %----------')
sum((g~= M6_Labels))/600


%%%%%%%%%%
load('M6_PreProc_testing.mat', 'M6_TImg', 'TestLabel');
%TESTSET_PROCESSING
M6L1_TVec=zeros(10000, 64*K);
for Img = 1:10000
    for row=1:8
        for col=1:8
            for layer=1:K
                %% get the function for the chosen RV
                func = f{row,col}(:,layer);
                
                %% M6L1_SVec is our training set expressed in the 256 dimension vector from
                %%% for each img, using function matrics to compute 256
                %%% value
                M6L1_TVec(Img, (row-1)*32+(col-1)*K+layer) = func(M6_TImg(Img,row,col));
            end
        end
    end
end
%%%%%%%%%%



%% predict on test set
[g,~]=predict(SVModel, M6L1_TVec);
disp('-------error in prediction----------')
sum((g~= TestLabel))
disp('-------percentage error in prediction in %----------')
sum((g~= TestLabel))/100

%%if a cross validation (10-fold) is needed
%%using the code below
%%CV = crossval(SVModel);
%%oosLoss = kfoldLoss(CV)
