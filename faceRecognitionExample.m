function faceRecognitionExample()
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

  % Número de faces com maior distinção consideradas
  numFaces = 20;
  
  for indiceImagem = 1:bancoImagens.numImagens
    % Imagem Escolhida
    imagemEscolhida = matrizImagens(:, indiceImagem);

    % Matriz com todas as imagens, menos a escolhida
    imagensRestantes = matrizImagens(:, [1:(indiceImagem - 1) (indiceImagem + 1):end]);

    %% Etapa de treinamento-----------------------------------------------------
    %% -------------------------------------------------------------------------
    
    % Média das imagens restantes
    mediaImagens = uint8(mean(imagensRestantes, 2));
    
    % Vetor linha em que todos os elementos são iguais a 1 e cujo
    % número de colunas é igual ao número de imagens restantes
    vetorAux = uint8(ones(1, size(imagensRestantes,2)));

    % Ao se multiplicar o vetor coluna que representa a média das
    % imagens pelo vetorAux, termos uma matriz com numImagens colunas
    % idênticas e iguais à face média
    % Em seguida, subtrai-se as imagens restantes da média,
    % obtendo-se as imagens deslocadas
    imagensRestantesDeslocadas = imagensRestantes - uint8(single(mediaImagens)*single(vetorAux));

    % Finding the correlation matrix.
    matrizCovarianca = single(imagensRestantesDeslocadas)'*single(imagensRestantesDeslocadas);

    % Cálculo dos autovetores.
    [autoVetores, autoValores] = eig(matrizCovarianca);

    autoVetores = single(imagensRestantesDeslocadas)*autoVetores;

    % Selecioando os autovetores correspondente aos numFaces maiores
    % autovalores
    autoVetores = autoVetores(:, end:-1:end - (numFaces - 1));

    % Calculando a assinatura de cada imagem
    % Cada linha da matriz "assinaturas" é a assinatura de uma imagem
    assinaturas = zeros(size(imagensRestantes, 2), numFaces);

    for i = 1:size(imagensRestantes, 2),
        assinaturas(i, :) = single(imagensRestantesDeslocadas(:, i))' * autoVetores;
    end

    %% Etapa de reconhecimento--------------------------------------------------
    %% -------------------------------------------------------------------------
    subplot(1, 2, 1);
    imshow(vetorColunaParaMatriz(imagemEscolhida, bancoImagens.alturaImagens, bancoImagens.larguraImagens), []);
    title('Looking for ...', 'FontWeight', 'bold', 'Fontsize', 16, 'color', 'red');

    % Subtraída da imagem escolhida a média das imagens
    imagemEscolhidaDeslocada = imagemEscolhida - mediaImagens;

    % Image escolhida projetada no espaço de autovetores
    imagemProjetada = single(imagemEscolhidaDeslocada)'*autoVetores;

    %  Vetor de distâncias entre a face escolhida e as outras
    dist = [];

    subplot(1, 2, 2);
    for i = 1:size(imagensRestantes, 2),
        % Calcula a distância entre a imagem atual e a imagem escolhida
        % através da norma euclidiana e adiciona ao vetor de distâncias
        dist = [dist, norm(assinaturas(i, :) - imagemProjetada, 2)];
    end

    % Determina a imagem mais próxima da imagem imagem escolhida
    [distMinima, indiceDistMinima] = min(dist);
    subplot(1, 2, 2);
    imshow(vetorColunaParaMatriz(imagensRestantes(:,indiceDistMinima), bancoImagens.alturaImagens, bancoImagens.larguraImagens), []);
    title('Found!', 'FontWeight', 'bold', 'Fontsize', 16, 'color', 'red');
    input("Continuar");
  end
end

% Converte vetor coluna para matriz
function matriz = vetorColunaParaMatriz(vetor, altura, largura)
  matriz = zeros(altura, largura);

  for i = 1:altura
      matriz(i, :) = vetor((i - 1)*largura + 1:i*largura);
  end
end