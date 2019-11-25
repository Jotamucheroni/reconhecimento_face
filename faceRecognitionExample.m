function faceRecognitionExample()
  bancoImagens.prefixo = 'subject';
  bancoImagens.extensao = 'gif';
  bancoImagens.alturaImagens = 243;
  bancoImagens.larguraImagens = 320;

  % Carrega imagens da pasta
  [matrizTreino, identidadeTreino] = carregarImagens('YalesFace/treinamento/', ...
                                                       bancoImagens.prefixo, ...
                                                       bancoImagens.extensao, ...
                                                       bancoImagens.alturaImagens, ...
                                                       bancoImagens.larguraImagens);
  [matrizTeste, identidadeTeste] = carregarImagens('YalesFace/teste/', ...
                                                       bancoImagens.prefixo, ...
                                                       bancoImagens.extensao, ...
                                                       bancoImagens.alturaImagens, ...
                                                       bancoImagens.larguraImagens);                                                    
  
  % Número de faces com maior distinção consideradas
  numFaces = 20;
  
  %% Etapa de treinamento-----------------------------------------------------
  %% -------------------------------------------------------------------------
  
  % Média das imagens restantes
  mediaImagens = uint8(mean(matrizTreino, 2));
  
  % Vetor linha em que todos os elementos são iguais a 1 e cujo
  % número de colunas é igual ao número de imagens restantes
  vetorAux = uint8(ones(1, size(matrizTreino,2)));

  % Ao se multiplicar o vetor coluna que representa a média das
  % imagens pelo vetorAux, termos uma matriz com numImagens colunas
  % idênticas e iguais à face média
  % Em seguida, subtrai-se as imagens restantes da média,
  % obtendo-se as imagens deslocadas
  imagensDeslocadas = matrizTreino - uint8(single(mediaImagens)*single(vetorAux));
  
  % Finding the correlation matrix.
  matrizCovarianca = single(imagensDeslocadas)'*single(imagensDeslocadas);

  % Cálculo dos autovetores.
  [autoVetores, autoValores] = eig(matrizCovarianca);
  autoVetores = single(imagensDeslocadas)*autoVetores;

  % Selecioando os autovetores correspondente aos numFaces maiores
  % autovalores
  autoVetores = autoVetores(:, end:-1:end - (numFaces - 1));

  % Calculando a assinatura de cada imagem
  % Cada linha da matriz "assinaturas" é a assinatura de uma imagem
  assinaturas = zeros(size(matrizTreino, 2), numFaces);

  for i = 1:size(matrizTreino, 2),
      assinaturas(i, :) = single(imagensDeslocadas(:, i))' * autoVetores;
  end
  
  %% Etapa de reconhecimento--------------------------------------------------
  %% -------------------------------------------------------------------------
  
  %  Vetor de distâncias entre a face escolhida e as outras
  dist = zeros(size(matrizTeste, 2), size(matrizTreino, 2));
  
  for indiceImagem = 1:size(matrizTeste, 2)
    % Imagem Escolhida
    imagemEscolhida = matrizTeste(:, indiceImagem);

    % Subtraída da imagem escolhida a média das imagens
    imagemEscolhidaDeslocada = imagemEscolhida - mediaImagens;

    % Image escolhida projetada no espaço de autovetores
    imagemProjetada = single(imagemEscolhidaDeslocada)'*autoVetores;

    for i = 1:size(matrizTreino, 2)
        % Calcula a distância entre a imagem atual e a imagem escolhida
        % através da norma euclidiana e adiciona ao vetor de distâncias
        dist(indiceImagem, i) = norm(assinaturas(i, :) - imagemProjetada, 2);
    end
  end
  
  %% Cálculo da matriz de confusão--------------------------------------------
  %% -------------------------------------------------------------------------
  
  % Verdadeiros positivos
  VP = 0;
  % Verdadeiros negativos
  VN = 0;
  % Falsos positivos
  FP = 0;
  % Falsos negativos
  FN = 0;
  % Normalização das distâncias pela distância máxima
  dist /= max(dist(:));
  % Limiar de distância para reconhecer a face
  limiar = 0.26;
  % Binarização das comparações a partir do limiar
  dist = dist <= limiar;
  for indiceImagem = 1:size(matrizTeste, 2)
    for i = 1:size(matrizTreino, 2)
      if identidadeTeste(indiceImagem) == identidadeTreino(i)
        if dist(indiceImagem, i)
          VP++;
        else
          FN++;
        end
      else
        if !dist(indiceImagem, i)
          VN++;
        else
          FP++;
        end
      end
    end
  end
  
  printf("Limiar = %0.10f\n", limiar)
  printf("VP = %d\n", VP);
  printf("VN = %d\n", VN);
  printf("FP = %d\n", FP);
  printf("FN = %d\n", FN);
  printf("Acurácia = %f\n", (VP + VN) / (VP + VN + FP + FN) * 100);
  taxaFP = FP / (FP + VN) * 100;
  taxaFN = FN / (FN + VP) * 100;
  printf("Taxa de FP = %f\n", taxaFP);
  printf("Taxa de FN = %f\n", taxaFN);
  printf("Diferença entre taxa de FP e de FN = %f\n", abs(taxaFP - taxaFN));
  precisao = VP / (VP + FP);
  revocacao = VP / (VP + FN);
  printf("Precisão = %f\n", precisao * 100);
  printf("Revocação = %f\n", revocacao * 100);
  printf("Especificidade = %f\n", VN / (VN + FP) * 100);
  printf("F1 = %f\n", 2 / (1 / precisao + 1 / revocacao) * 100);
end

% Converte vetor coluna para matriz
function matriz = vetorColunaParaMatriz(vetor, altura, largura)
  matriz = zeros(altura, largura);

  for i = 1:altura
      matriz(i, :) = vetor((i - 1)*largura + 1:i*largura);
  end
end