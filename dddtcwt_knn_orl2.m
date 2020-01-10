clear all;
close all;
clc;

 my_path ='C:\Users\deepanshu\Desktop\research papers\PCA_FRS_ORL_V_3.3\PCA_FRS_ORL_V_3.3\Database\ORL\';
 no_t_img_p_sub=7;
 subDim=40;         % dimension= No_of_subject * no_t_img_p_sub -1
 total_image_p_subject=10;
 No_of_subjects=40;
 
 
 Create_Train_Test_Data_List(my_path, no_t_img_p_sub,total_image_p_subject,No_of_subjects);
 
 % Creating training images space
fprintf('Loading training images list from AT&T face database\n')
load TrainData_List;
load m;     % Number of rows in the images of database
load n;     % Number of columns in the images of database

fprintf('Creating training images space from AT&T face database\n')
dim = length(TrainData_List);

if(dim<subDim)
    subDim = dim-1;
end


imSpace = zeros (m*n, dim);       % Memory allocated

for i = 1 : dim
    im = imread ( [my_path char(TrainData_List(i,:))] ) ;
    im=imcrop(im, [0 0 128 128]);
    imSpace(:, i) = reshape(im, m*n, 1);
end;
save imSpace imSpace;

% +++++++++++++++++ Calculating mean face from training images

fprintf('Zero mean\n')
my_psi = mean(double(imSpace'))';
save my_psi my_psi;

% Zero mean
zeroMeanSpace = zeros(size(imSpace));
for i = 1 : dim
    zeroMeanSpace(:, i) = double(imSpace(:, i)) - my_psi;
end;
save zeroMeanSpace zeroMeanSpace;
clear imSpace;

% % % % % Applying Dual tree complex wavelet transform


for i = 1 : dim
    a = zeroMeanSpace(:, i) ;
    d = reshape(a,[m n]);
    wt = dddtcwt(d,2);
    [y z] = size(wt);
    DTCWTzeroMeanSpace(:, i)= reshape(wt, y*z, 1);
    
end;

save DTCWTzeroMeanSpace DTCWTzeroMeanSpace; 


% PCA
fprintf('PCA\n')
L=DTCWTzeroMeanSpace'*DTCWTzeroMeanSpace;
[eigVecs, eigVals] = eig(L);
diagonal = diag(eigVals);
[diagonal, index] = sort(diagonal);
index = flipud(index);
 
pcaEigVals = zeros(size(eigVals));
for i = 1 : size(eigVals, 1)
    pcaEigVals(i, i) = eigVals(index(i), index(i));
    pcaEigVecs(:, i) = eigVecs(:, index(i));
end;

pcaEigVals = diag(pcaEigVals);
pcaEigVals = pcaEigVals / (dim-1);
pcaEigVals = pcaEigVals(1 : subDim);        % Retaining only the largest subDim ones

pcaEigVecs = DTCWTzeroMeanSpace * pcaEigVecs;    % Turk-Pentland trick (part 2)

% save pcaEigVals pcaEigVals;

% Normalisation to unit length
fprintf('Normalising\n')
for i = 1 : dim
    pcaEigVecs(:, i) = pcaEigVecs(:, i) / norm(pcaEigVecs(:, i));
end;

% Dimensionality reduction. 
fprintf('Creating lower dimensional subspace\n')
w = pcaEigVecs(:, 1:subDim);
save w w;
clear w;

%%%%%%%%%%%%%%%%%%%%%%PCA PROJECTION of DWT zero mean space



load w;
load  DTCWTzeroMeanSpace;

fprintf('Projecting all images of this training subset'); 
fprintf('onto a new lower dimensional subspace\n');

PCAProjection = w' * DTCWTzeroMeanSpace;
clear DTCWTzeroMeanSpace;
save PCAProjection PCAProjection;

%%% Face recognition using DTCWT



load m;
load n;% To get the dimension of an image from ATT
load w;% To get the lower dimesion subspace
% % % % % load TestData_List;
% % % % % load TrainData_List;
load my_psi;

load TestData_List;
load TrainData_List;

% % % % % 
% % % % % % Step 1.....load all test images
tic
% % % % % 
 dim1 = length(TestData_List);
 dim2 = length(TrainData_List);
 imageSpaceTest1 = zeros (m*n, dim1);       % Memory allocated
 
 for i = 1 : dim1
    im = imread ( [my_path char(TestData_List(i,:))] ) ;
  im=imcrop(im, [0 0 128 128]);
  imageSpaceTest1(:, i) = reshape(im, m*n, 1);
  end;
% % % % % 
 save imSpaceTest1 imageSpaceTest1;
 

 % % %STEP  2
 
 zeroMeanSpaceTest1 = zeros(size(imageSpaceTest1));
 for i = 1 : dim1
     zeroMeanSpaceTest1(:, i) = double(imageSpaceTest1(:, i)) - my_psi;
 end;
 save zeroMeanSpaceTest1 zeroMeanSpaceTest1;
 
 
 
 %%%%%%%Create  DTCWTzeroMeanSpaceTest1
 
 % % % % % Applying Dual tree complex Wavelet transform on test images


for i = 1 : dim1
    a = zeroMeanSpaceTest1(:, i) ;
    d = reshape(a,[m n]);
    wt = dddtcwt(d,2);
    [y z] = size(wt);
    DTCWTzeroMeanSpaceTest(:, i)= reshape(wt, y*z, 1);
    
end;

save DTCWTzeroMeanSpaceTest DTCWTzeroMeanSpaceTest; 
clear zeroMeanSpaceTest1;


%%%%% Calculate Test Projection matrix
TestProjection = w'* DTCWTzeroMeanSpaceTest;

save TestProjection TestProjection;
 J=[1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3 3 3 3 3 4 4 4 4 4 4 4 5 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 8 8 8 8 8 8 8 9 9 9 9 9 9 9 10 10 10 10 10 10 10 11 11 11 11 11 11 11 12 12 12 12 12 12 12 13 13 13 13 13 13 13 14 14 14 14 14 14 14 15 15 15 15 15 15 15 16 16 16 16 16 16 16 17 17 17 17 17 17 17 18 18 18 18 18 18 18 19 19 19 19 19 19 19 20 20 20 20 20 20 20 21 21 21 21 21 21 21 22 22 22 22 22 22 22 23 23 23 23 23 23 23 24 24 24 24 24 24 24 25 25 25 25 25 25 25 26 26 26 26 26 26 26 27 27 27 27 27 27 27 28 28 28 28 28 28 28 29 29 29 29 29 29 29 30 30 30 30 30 30 30 31 31 31 31 31 31 31 32 32 32 32 32 32 32 33 33 33 33 33 33 33 34 34 34 34 34 34 34 35 35 35 35 35 35 35 36 36 36 36 36 36 36 37 37 37 37 37 37 37 38 38 38 38 38 38 38 39 39 39 39 39 39 39 40 40 40 40 40 40 40];
 Mdl = fitcknn(PCAProjection',J,'NumNeighbors',1,... 
    'NSMethod','exhaustive','Distance','euclidean');
rng(0); % For reproducibility
CVKNNMdl = crossval(Mdl);
classError = kfoldLoss(CVKNNMdl)
[label,score,cost] = predict(Mdl,TestProjection')
G=[1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9 10 10 10 11 11 11 12 12 12 13 13 13 14 14 14 15 15 15 16 16 16 17 17 17 18 18 18 19 19 19 20 20 20 21 21 21 22 22 22 23 23 23 24 24 24 25 25 25 26 26 26 27 27 27 28 28 28 29 29 29 30 30 30 31 31 31 32 32 32 33 33 33 34 34 34 35 35 35 36 36 36 37 37 37 38 38 38 39 39 39 40 40 40]   
%classification accuracy
B=G';
D=TestProjection';
 recograte=length(find(label==B))/size(D,1)*100
% % % % % %Step4 
% % %  A

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%Calculating euclidean distance between Test projection matrix and PCA
%%projection matrix
dim1 = length(TestData_List);
 dim2 = length(TrainData_List);

%for j=1:dim1
 %   for i=1:dim2
  %      c = (TestProjection(:,j)-PCAProjection(:,i)).^2;
   %     distmetric(i,j)= (subDim*mean(c))^0.5;
        
        % c = (TestProjection(:,j)-PCAProjection(:,i));
         % distmetric(i,j)= sum(abs(c));
    %end    
 %end

% % %   B
%%% distance metric calculated through function pdist2
%distmetric= pdist2(PCAProjection',TestProjection','euclidean');
%%%%%%
% Minimum of distance metric will be calculated of as below 

% for p=1:dim1 %%% dim1 = length(TestData_List);

 %    [val(:,p),index(:,p)]=(min(distmetric(:,p)));
  %   matchvalue(:,p)=ceil(index(:,p)/no_t_img_p_sub);

 %end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%%%% %%%%%%%%%%%%%%%%%
% % % % C
% %  distance metric calculated by Similarity Index Matrix .....
% % %  The value should be between 1 and -1
% %  if 1 then perfectly matched
% %  if -1 then perfectly unmatched
% for similarity index corr2 fucntion cannot work because it requires
% matrices of same size

% % B=TestProjection;
% % A=PCAProjection;
% % 
% % for j=1:dim1
% %     
% %     for i=1:dim2
% %         xiyi = B(:,j).*A(:,i);
% %         num = sum (xiyi);
% %         xisqr = (A(:,i)).^2;
% %         yisqr = (B(:,j)).^2;
% %         denom = (sum(xisqr)* sum(yisqr))^0.5;
% %      distmetric(i,j) =  -(num / denom);
% %     end    
% % end
% % 
% % for p=1:dim1 %%% dim1 = length(TestData_List);
% % 
% %      [val(:,p),index(:,p)]=(min(distmetric(:,p)));
% %      matchvalue(:,p)=ceil(index(:,p)/no_t_img_p_sub);
% % 
% % end
 

%save distmetric distmetric;
%save index index;
%save val val;
%save matchvalue matchvalue;


%%%%
% % % % %Step 5 %%
% % dim1 = length(TestData_List);
% %  dim2 = length(TrainData_List);



%matchcount=0;
%unmatchcount=0;
 
% % % % % Step 6
%remianing_test_images=total_image_p_subject-no_t_img_p_sub;% How much remaining test images are left
%% Earlier concept

% for j=1:No_of_subjects
%      for k=1:remianing_test_images 
% 
%     
%          matchvalue(:,remianing_test_images*(j-1)+k)=ceil(index(:,remianing_test_images*(j-1)+k)/no_t_img_p_sub);
%      
%     end
%  end
 
 % % dim1 = length(TestData_List);
% %  dim2 = length(TrainData_List);

%for j=1:No_of_subjects
 %    for k=1:remianing_test_images
  %       if matchvalue(:,remianing_test_images*(j-1)+k)== j
   %         matchcount=matchcount+1;
    %     else
     %      unmatchcount=unmatchcount+1;     
      %   end
     %end
%end

%% Calculating Recog Rate
 %recograte = (matchcount/dim1)*100

%errorrate=(unmatchcount/dim1)*100;
toc