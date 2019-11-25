function imagensFormatadas = carregarImagens(pasta, prefixo, extensaoImagens, numImagens, altura, largura)
  % Número de pixels de cada imagem = altura * largura
  imagensFormatadas = zeros(altura * largura, numImagens);
  nomeImagens = glob([pasta "*" extensaoImagens]);

  for i=1:size(nomeImagens, 1)
    imagensFormatadas(:, i) = matrizParaVetorColuna(imread(nomeImagens{i}));
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