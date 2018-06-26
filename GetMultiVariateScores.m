clear;
train_num = 200
load ('M6_PreProc.mat');

X = M6_RImg; % Set X as the subimages for simplicity

K = 8; %Take 8 layers (eigenmodes) of score functions

Xcard= squeeze(max(X)); %Cardinalities of 8X8 = 64 input areas

f = cell(64,1); %f{area} is (x, layer) function


%%%%%%%%% Initialize zero-mean score functions
for layer = 1:K
    for area=1:64
        f{area}(:,layer) = randn(Xcard(area), 1);
        tmp_fxImg(:,area) = f{area}(X(:,area),layer);
    end;
    tmp = reshape(tmp_fxImg,[train_num*64,1]);
    fmean = sum(tmp)/(train_num*64);
    for area=1:64
        f{area}(:,layer) = f{area}(:,layer) - fmean;
    end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

number_iteration = 15; % Run 15 iteration for ACE algorithm

% Check the correlation between scores at each iteration and layer
correlation = zeros(number_iteration,K); 


for layer=1:K
    for iteration = 1:number_iteration
        tF = f;
        %%%%%% Compute the conditional expectations
        for area = 1:64
            for xvalue=1:Xcard(area)
                subseq = (X(:,area) == xvalue);
                subseqL = sum(subseq);
                offset = zeros(subseqL, 1); 
                for i=1:64
                    if (i~=area)
                        offset = offset + tF{i}(X(subseq, i), layer);
                    end;
                end;
                f{area}(xvalue, layer) = sum(offset) / subseqL;
            end; % xvalue
        end; % area
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% Orthogalization: Gram-Schmidt procedure.
        inner_product = zeros(K,1);
        for i = 1:layer-1
            fxImg_layer = zeros(train_num,64);
            fxImg_i = zeros(train_num,64);
            for area = 1:64
                fxImg_layer(:,area) = f{area}(X(:,area),layer);
                fxImg_i(:,area) = f{area}(X(:,area),i);
            end;
            for area = 1:64
                inner_product(i) = inner_product(i) + sum(fxImg_layer(:,area).*fxImg_i(:,area))/train_num;
            end;
        end; 
        for i = 1:layer-1
            for area = 1:64
                f{area}(:,layer) = f{area}(:,layer) - f{area}(:,i).*inner_product(i);
            end;
        end;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%% Normalization: zero-mean and unit-variance
        fxImg = zeros(train_num,64);
        for area = 1:64
            fxImg(:,area) = f{area}(X(:,area),layer);
        end;
        tmp = reshape(fxImg,[train_num*64,1]);
        fmean = sum(tmp)/(train_num*64);
        fnorm = sqrt(sum(tmp.^2)/train_num);
        for area = 1:64
            f{area}(:,layer) = (f{area}(:,layer)-fmean)./fnorm;
        end;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%% Print the iteration, layer
        %%%%%%%%% Check the correlation
        for area = 1:64
            fxImg(:,area) = f{area}(X(:,area),layer);
        end;
        %tmp = reshape(fxImg,[train_num*64,1]);
        %mean = sum(tmp)/train_num
        %norm = sqrt(sum(tmp.^2)/train_num)
        %correlation(iteration) = 0;
        for i = 1:63
            for j = i+1:1:64
                correlation(iteration,layer) = correlation(iteration,layer) + sum(fxImg(:,i).*fxImg(:,j))/train_num;
            end;
        end;  
        layer
        iteration
        correlation(iteration,layer)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    end; % iteration
end; % layer
    
save('MVC_score_layer1-8_iteration15.mat' , 'f','correlation');