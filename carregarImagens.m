function [imagensFormatadas, identidadeImagens] = carregarImagens(pasta, prefixo, extensaoImagens, altura, largura)
  % Nomes de todas as imagens da pasta informada
  nomeImagens = glob([pasta "*" extensaoImagens]);
  % Número de pixels de cada imagem = altura * largura
  imagensFormatadas = zeros(altura * largura, size(nomeImagens, 1));
  % Identidade do sujeito que aparece na imagem
  identidadeImagens = zeros(1, size(nomeImagens, 1));

  for i=1:size(nomeImagens, 1)
    imagensFormatadas(:, i) = matrizParaVetorColuna(imread(nomeImagens{i}));
    identidadeImagens(i) = uint8(sscanf(nomeImagens{i}, [pasta prefixo "%d"]));
  end
  
  % Converte para inteiro sem sinal de 8 bits para poupar memória
  imagensFormatadas = uint8(imagensFormatadas);
end

% Converte matriz para vetor coluna
function vetorColuna = matrizParaVetorColuna(matriz)
  [altura, largura] = size(matriz);

  for i = 1:altura
      vetorColuna((i - 1)*largura + 1:i*largura) = matriz(i, :);
  end;

  vetorColuna = vetorColuna';
end