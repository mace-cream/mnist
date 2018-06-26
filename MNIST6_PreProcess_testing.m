clear;

% This does the same thing as 'MNIST6_PreProcess.m' but on the testing data
% The preprocessed testing image is saved at 'M6_PreProc_testing.mat'
% for further processing

fTestImg=fopen('t10k-images-idx3-ubyte', 'r');

temp=fread(fTestImg,16);

TestImg=zeros(10000, 28,28);

for i=1:10000
    TestImg(i,:,:)=fread(fTestImg, [28,28]);
end;

fclose(fTestImg);

fTestLabel=fopen('t10k-labels-idx1-ubyte', 'r');

temp=fread(fTestLabel, 8);
TestLabel=fread(fTestLabel, 10000);
fclose(fTestLabel);


% generate M6L1_TVec to record all the scores.  
load('M6_PreProc.mat','Alphabet');


pos=[1,4,7,10,13, 16, 19, 22];
k=4;
M6_TImg=zeros(10000, 8,8); % store the area values 



for Img=1:10000
    for i=1:8,
        ipos=pos(i);
        for j=1:8
            jpos=pos(j);
            subimg=reshape((TestImg(Img, ipos:ipos+5, jpos:jpos+5)>40), [36,1]);
            Matimg=repmat(subimg, [1,size(Alphabet{i,j},2)]);
            Matimg=Matimg-Alphabet{i,j};
            Matimg=Matimg.*Matimg;
            dist2= sum(Matimg);
            [~,areavalue]=min(dist2);
            M6_TImg(Img, i, j)=areavalue;
        end;
    end;
end;

save('M6_PreProc_testing.mat', 'M6_TImg', 'TestLabel');
