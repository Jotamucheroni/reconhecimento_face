function done = faceRecognitionExample()
  bancoImagens.pasta = 'ATeT/';
  bancoImagens.prefixo = 's';
  bancoImagens.extensao = 'pgm';
  bancoImagens.numImagens = 400;
  bancoImagens.alturaImagens = 112;
  bancoImagens.larguraImagens = 92;

  % Carrega imagens da pasta
  matrizImagens = carregarImagens(bancoImagens.pasta, ...
                                       bancoImagens.prefixo, ...
                                       bancoImagens.extensao, ...
                                       bancoImagens.numImagens, ...
                                       bancoImagens.alturaImagens, ...
                                       bancoImagens.larguraImagens);

  % Sorteia uma imagem
  indiceImagemAleatoria = round(bancoImagens.numImagens*rand());
  imagemAleatoria = matrizImagens(:, indiceImagemAleatoria);

  % The "otherFacesDatabaseMatrix" contains the rest of the 399 images.
  imagensRestantes = matrizImagens(:, [1:(indiceImagemAleatoria - 1) (indiceImagemAleatoria + 1):end]);

  % Number of signatures used for each image.
  amountOfSignaturesUsed = 20;

  % Subtracting the mean from the "otherFacesDatabaseMatrix". A temporary
  % matrix used to help on finding the mean of all images in the
  % "otherFacesDatabaseMatrix".
  temporaryMatrix = uint8(ones(1, size(imagensRestantes,2)));

  % The "meanOfAllImages" is a matrix holding the mean of all images.
  meanOfAllImages = uint8(mean(imagensRestantes, 2));

  % The "meanRemovedFacesDatabaseMatrix" is the "otherFacesDatabaseMatrix"
  % with its mean removed. 
  meanRemovedFacesDatabaseMatrix = imagensRestantes - uint8(single(meanOfAllImages)*single(temporaryMatrix));

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
  imageSignatures = zeros(size(imagensRestantes, 2), amountOfSignaturesUsed);

  for i = 1:size(imagensRestantes, 2),
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
  imshow(vetorColunaParaMatriz(imagemAleatoria, bancoImagens.alturaImagens, bancoImagens.larguraImagens), []);
  title('Looking for ...', 'FontWeight', 'bold', 'Fontsize', 16, 'color', 'red');

  subplot(1, 2, 2);
  % Always remember to subtract the mean!
  randomImageWithoutMean = imagemAleatoria - meanOfAllImages;

  % Projecting the "randomImageWithoutMean" onto the eigenvectors space.
  imageProjection = single(randomImageWithoutMean)'*V;

  % This "z" here is used to keep track of the differences between the
  % correct face and the others.
  z = [];

  for i = 1:size(imagensRestantes, 2),
      % The norm of a matrix is a scalar that gives some measure of the
      % magnitude of the elements of the matrix. It is interesting to use it
      % together with the PCA technique becasue when "A" is a matrix, then
      % norm(A) returns the largest SINGULAR VALUE of A (in other words, it
      % returns "max(svd(A))").
      z = [z, norm(imageSignatures(i, :) - imageProjection, 2)];
      
      % To speed things, we only show one at each 20 images during the search
      % for the correct face.
      if (rem(i, 20) == 0),
          imshow(vetorColunaParaMatriz(imagensRestantes(:, i), bancoImagens.alturaImagens, bancoImagens.larguraImagens), []);
      end;
      drawnow;
  end

  [a,i] = min(z);
  subplot(1, 2, 2);
  imshow(vetorColunaParaMatriz(imagensRestantes(:,i), bancoImagens.alturaImagens, bancoImagens.larguraImagens), []);
  title('Found!', 'FontWeight', 'bold', 'Fontsize', 16, 'color', 'red');

  done = 1;
end

% Converte vetor coluna para matriz
function matriz = vetorColunaParaMatriz(vetor, altura, largura)
  matriz = zeros(altura, largura);

  for i = 1:altura
      matriz(i, :) = vetor((i - 1)*largura + 1:i*largura);
  end
end