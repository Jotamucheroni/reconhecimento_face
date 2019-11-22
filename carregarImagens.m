function imagensFormatadas = carregarImagens(pasta, prefixo, extensaoImagens, numImagens, altura, largura)
  % Carregas as imagens uma única vez
  persistent imagensCarregadas;
  persistent matrizFaces;

  % Número de pixels de cada imagem = altura * largura
  matrizFaces = zeros(altura * largura, numImagens);
  % Verifica se as imagens não foram ainda carregadas
  if (isempty(imagensCarregadas))
      coluna = 1;
      for i = 1:40
          for j = 1:10      
              vetorColuna = matrizParaVetorColuna(imread([pasta prefixo num2str(i) '-' num2str(j) '.' extensaoImagens]));
              matrizFaces(:, coluna) = reshape(vetorColuna, size(vetorColuna, 1)*size(vetorColuna, 2), 1);
              coluna = coluna + 1;
          end
      end
      
      % Converte para inteiro sem sinal de 8 bits para poupar memória
      matrizFaces = uint8(matrizFaces);
      % Indica que as imagens já foram carregadas
      imagensCarregadas = 1;
  end

  imagensFormatadas = matrizFaces;
end

% Converte matriz para vetor coluna
function vetorColuna = matrizParaVetorColuna(matriz)
  [altura, largura] = size(matriz);

  for i = 1:altura
      vetorColuna((i - 1)*largura + 1:i*largura) = matriz(i, :);
  end;

  vetorColuna = vetorColuna';
end