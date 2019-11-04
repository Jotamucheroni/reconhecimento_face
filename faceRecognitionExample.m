function done = faceRecognitionExample()

% *************************************************************************
% Face recognition using PCA.
% 
% This algorithm uses the eigenface system (based on the Pricipal Component
% Analysis, PCA) to recognize faces. But first it is necessary to download
% the AT & T Face Database (the one we will use to perform our experiment).
% Do a little googling and it will be one of the first results. It contains
% 400 pictures of 40 subjects (each subject was photographed ten times).

% *************************************************************************
% Initialization **********************************************************
% 
% Loading the database into a matrix called "faceDatabaseMatrix".
faceDatabaseMatrix = carregarImagens(400, 112, 92);

% We randomly pick an image from our database and use the rest of the
% images for training (so, the training is done on 399 pictures). We later
% use the randomly selectted picture to test the algorithm.
randomImageIndex = round(400*rand(1,1));

% The "randomImage" now contains the image we later on will use to test the
% algorithm.
randomImage = faceDatabaseMatrix(:, randomImageIndex);

% The "otherFacesDatabaseMatrix" contains the rest of the 399 images.
otherFacesDatabaseMatrix = faceDatabaseMatrix(:, [1:randomImageIndex - 1 randomImageIndex + 1:end]);

% Number of signatures used for each image.
amountOfSignaturesUsed = 20;

% Subtracting the mean from the "otherFacesDatabaseMatrix". A temporary
% matrix used to help on finding the mean of all images in the
% "otherFacesDatabaseMatrix".
temporaryMatrix = uint8(ones(1, size(otherFacesDatabaseMatrix,2)));

% The "meanOfAllImages" is a matrix holding the mean of all images.
meanOfAllImages = uint8(mean(otherFacesDatabaseMatrix, 2));

% The "meanRemovedFacesDatabaseMatrix" is the "otherFacesDatabaseMatrix"
% with its mean removed. 
meanRemovedFacesDatabaseMatrix = otherFacesDatabaseMatrix - uint8(single(meanOfAllImages)*single(temporaryMatrix));

% *************************************************************************
% Calculating the eigenvectors of the correlation matrix ******************
% 
% We are picking only the "amountOfSignaturesUsed" eigenfaces over all of
% the 400 eigenfaces.

% Finding the correlation matrix.
correlationMatrix = single(meanRemovedFacesDatabaseMatrix)'*single(meanRemovedFacesDatabaseMatrix);

% Computing its eigenvectors.
[V, D] = eig(correlationMatrix);

V = single(meanRemovedFacesDatabaseMatrix)*V;

% Now pick the eigenvectors corresponding to the "amountOfSignaturesUsed"
% largest eigenvalues. 
V = V(:, end:-1:end - (amountOfSignaturesUsed - 1));

% Calculating the signature for each image.
% To speed things and help Matlab to deal with memory, we have again
% preallocated a matrix to hold the data. Each row in the matrix
% "imageSignatures" is the signature for one image.
imageSignatures = zeros(size(otherFacesDatabaseMatrix, 2), amountOfSignaturesUsed);

for i = 1:size(otherFacesDatabaseMatrix, 2),
    % Again, each row in the "imageSignatures" matrix is the signature for
    % one image.
    imageSignatures(i, :) = single(meanRemovedFacesDatabaseMatrix(:, i))'*V;
end

% *************************************************************************
% Performing the recognition task *****************************************
% 
%  Now, we run the algorithm and see if we can correctly recognize the
%  face.
subplot(1, 2, 1);
imshow(columnVectorToMatrixConversion(randomImage), []);
title('Looking for ...', 'FontWeight', 'bold', 'Fontsize', 16, 'color', 'red');

subplot(1, 2, 2);
% Always remember to subtract the mean!
randomImageWithoutMean = randomImage - meanOfAllImages;

% Projecting the "randomImageWithoutMean" onto the eigenvectors space.
imageProjection = single(randomImageWithoutMean)'*V;

% This "z" here is used to keep track of the differences between the
% correct face and the others.
z = [];

for i = 1:size(otherFacesDatabaseMatrix, 2),
    % The norm of a matrix is a scalar that gives some measure of the
    % magnitude of the elements of the matrix. It is interesting to use it
    % together with the PCA technique becasue when "A" is a matrix, then
    % norm(A) returns the largest SINGULAR VALUE of A (in other words, it
    % returns "max(svd(A))").
    z = [z, norm(imageSignatures(i, :) - imageProjection, 2)];
    
    % To speed things, we only show one at each 20 images during the search
    % for the correct face.
    if (rem(i, 20) == 0),
        imshow(columnVectorToMatrixConversion(otherFacesDatabaseMatrix(:, i)), []);
    end;
    drawnow;
end

[a,i] = min(z);
subplot(1, 2, 2);
imshow(columnVectorToMatrixConversion(otherFacesDatabaseMatrix(:,i)), []);
title('Found!', 'FontWeight', 'bold', 'Fontsize', 16, 'color', 'red');

done = 1;

% *************************************************************************
% Function "columnVectorToMatrixConversion". ******************************
function matrix = columnVectorToMatrixConversion(vector)
%
% Function designed to cope with converting a column vector matrix to an
% image matrix. Be careful on using this function because the final matrix 
% size is hardcoded to be 92 by 112 (this is the size of the faces in the
% AT & T Faces Database).

height = 112;
width = 92;

matrix = zeros(height, width);

for i = 1:height,
    matrix(i, :) = vector((i - 1)*width + 1:i*width);
end;