clear

% This preprocess includes the follows

% 1. read training image file, putin M6_OImg(60000, 28,28);
%     and labels in M6_Labels(60000);
% 2. Quantize each pixel to binary with threshold 40;
% 3. divide each image into 6x6 sub-images, at locations 
%       i, j = [1,4,7,10, 13, 16, 19, 22]
% 4. For each of the 8x8 subimage location, go through all the images
%    to find all the possible values in 2^36, quantize sub-images for
%    Hamming distance <=3. 
% 5. Store all the images in M6_RImg(60000, 8,8) with values as the indices
%    within each alphabet
% 6. Store Alphabet{8,8}, each position an array of [:, 36] possible values

train_num = 200
% open image file 
fimg = fopen("train-images-idx3-ubyte","r");

% read 16 bytes head to nothing
temp=fread(fimg, 16);

% room for train_num images, each with size 28 x 28
M6_OImg=zeros(train_num, 28, 28);

for i = 1:train_num,
M6_OImg(i, :,:) = fread(fimg, [28,28]);
end;

fclose(fimg);

% open label file
flab=fopen('train-labels-idx1-ubyte','r');
%discard the first 8 bytes
temp=fread(flab, 8);
%read "train_num" labels;
M6_Labels=fread(flab, train_num);
fclose(flab);

%create an alphabet for each position
% each cell element has a list of 16x1 patterns that are seen at this
% position

Alphabet=cell(8);

M6_RImg=zeros(train_num,8,8);

pos=[1,4,7,10,13, 16, 19, 22];

for i =1:8
    ipos=pos(i);
    for j=1:8
        jpos=pos(j);
        
        Alphabet{i,j}=[]; % store as column vectors
        
        for Img=1:train_num, % for every image
            
            subimg=reshape((M6_OImg(Img, ipos:ipos+5, jpos:jpos+5)>40), [36,1]);
            %return col
            currentAlphabetSize=size(Alphabet{i,j}, 2);
            if (currentAlphabetSize==0)
                %if empty
                % work on the empty case separately due to laziness
                Alphabet{i,j}=[Alphabet{i,j}, subimg];
                M6_RImg(Img, i,j) = 1;
            else
                % computed Hamming distance to all recorded patterns
                mimg=repmat(subimg, [1, currentAlphabetSize]);
                % the difference between this image and each all formal image
                % on same area on each pixals
                diff=(mimg~=Alphabet{i,j});  
                % sum difference between this image and all formal image
                distance=sum(diff);  
                [mdis, loc]=min(distance);
                if (mdis<=3)
                    % loc is label if mdis<=3 ,use formal label
                    M6_RImg(Img, i,j)=loc;
                else
                    % if mdis>3 ,create new label and add this to alphabat
                    Alphabet{i,j}=[Alphabet{i,j}, subimg];
                    % new label
                    M6_RImg(Img, i,j) = currentAlphabetSize+1;
                end;
            end;
        end;
    end;
end;


%'M6_RImg' ----- "train_num"*8*8matrics 'Alphabet' ----vector  D = 36

save('M6_PreProc.mat', 'M6_OImg', 'M6_Labels', 'Alphabet', 'M6_RImg');
% X_1,\dots,X_64 X_i\in {1,\dots, |\mathcal{X}|}

