# HRSID: detecção e segmentação de navios

Este documento descreve [hrsid_mask_rcnn_transfer_learning.ipynb](../notebooks/hrsid_mask_rcnn_transfer_learning.ipynb), independente do experimento [DeepPCB](parametros_treinamento.md). Consulte o [ambiente compartilhado](environment.md) para instalação e execução.

## Dataset e preparação

O [HRSID oficial](https://github.com/chaozhong2010/HRSID) contém recortes de imagens de radar de abertura sintética (SAR), com navios em regiões costeiras e oceânicas. As imagens de 800 × 800 pixels possuem aparência monocromática, ruído *speckle* e variações de contexto. As anotações COCO descrevem caixas e máscaras por instância; o notebook trata uma classe de interesse, navio, além do fundo.

O arquivo `MyDrive/vision-transfer-learning/dataset/HRSID_jpg.zip` pode estar armazenado no Drive, mas o notebook atual **não lê nem extrai o ZIP**. Se houver apenas o arquivo compactado, extraia-o previamente no Drive, preservando imagens e anotações, e configure o caminho da pasta extraída. Nenhum download do dataset é necessário quando ele já está disponível.

```text
MyDrive/vision-transfer-learning/dataset/
├── DeepPCB/PCBData/
├── HRSID_jpg.zip              # arquivo armazenado, se disponível
└── HRSID/                     # pasta extraída usada pelo notebook
    └── .../HRSID_JPG/
        ├── annotations/       # train2017.json, test2017.json, etc.
        └── JPEGImages/
```

Com `CONFIG['dataset_dir']=''`, o notebook procura `data/HRSID/` ou `data/HRSID_jpg/` localmente; no Colab procura `MyDrive/vision-transfer-learning/dataset/HRSID/` ou `HRSID_jpg/`. Também aceita um caminho local, disco externo ou Drive montado/sincronizado, absoluto ou relativo à raiz do projeto. Descobre recursivamente as anotações e a pasta de imagens. Não copia os arquivos. `results_dir` pode mudar o destino dos resultados, que não pode ficar dentro do dataset.

A leitura converte a imagem para RGB e tensor em `[0, 1]`; dados originalmente monocromáticos tornam-se três canais. O modelo aplica sua normalização interna. Caixas COCO XYWH são convertidas para XYXY; polígonos e RLE são decodificados em máscaras por instância. Caixas degeneradas e máscaras vazias são descartadas em conjunto; imagens sem navios são mantidas como exemplos negativos. No treino, há espelhamento horizontal conjunto de imagem, caixas e máscaras, com probabilidade 0,5. A geometria vista de cima motiva o teste dessa transformação, sem garantir ganho no domínio SAR.

### Verificação de integridade

O notebook identifica arquivos de treino e teste, conta imagens, anotações e categorias, verifica arquivos ausentes, IDs e nomes duplicados, caixas inválidas ou fora da imagem e estrutura das segmentações. As dimensões declaradas são resumidas; uma amostra de até `sample_images=200` imagens por split é aberta para verificar formato, modo, dimensões reais e máscaras decodificadas. Os problemas encontrados são registrados em `reports/integrity_<split>.csv`; a verificação de pixels não cobre todos os arquivos. Anotações auxiliares, como listas costeiras e oceânicas, são listadas, não auditadas como splits.

A referência publicada indica 5.604 imagens e 16.951 navios; esse número de navios corresponde ao arquivo combinado. Em uma auditoria anterior da cópia oficial, `train2017.json` e `test2017.json` continham, respectivamente, 3.642/1.962 imagens e 11.047/5.922 anotações: a soma, 16.969, difere do combinado. Essas contagens são contexto, não resultados verificados nesta revisão; use o relatório da cópia executada.

### Divisão e vazamento

Quando presentes, os splits oficiais de treino e teste são preservados. A validação é extraída do treino por cena, identificada pelo prefixo antes de `_`, visando `val_fraction=0.15` das imagens. Cenas têm tamanhos diferentes, de modo que a fração é aproximada. O notebook informa as cenas compartilhadas entre splits e grava `reports/split.csv`. Se não houver dois grupos de cenas, a função recorre à divisão por imagem: nesse caso não há garantia de isolamento de cena. Sem os dois splits oficiais, deriva treino, validação e teste das anotações disponíveis; essa alternativa não é o teste oficial.

Na cópia auditada anteriormente, treino e teste oficiais compartilhavam cenas. A separação por cena evita a mesma fonte entre treino e validação quando há grupos suficientes, mas o teste oficial pode avaliar recortes novos de cenas já vistas. Isso limita a interpretação para cenas ou sensores independentes. `limit_images` amostra cada split após a divisão para ensaios rápidos; os resultados limitados não representam o conjunto completo.

## Modelo e transferência

O notebook usa o [`maskrcnn_resnet50_fpn_v2`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) oficial com `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`. O ResNet-50 extrai características e a Feature Pyramid Network (FPN) fornece representações em várias escalas. A Region Proposal Network (RPN) propõe regiões; RoI Align as amostra para as cabeças que classificam e refinam caixas e para a cabeça que prevê uma máscara por instância.

O notebook troca `FastRCNNPredictor` e `MaskRCNNPredictor` por preditores de duas classes: 0 para fundo e 1 para navio. As demais partes usam inicialmente os pesos COCO. Entrada: lista de tensores RGB e alvos com caixas, rótulos, máscaras, `image_id`, `area` e `iscrowd`. Saída em inferência: caixas, classes, scores e máscaras por imagem. Bordas e padrões espaciais aprendidos em fotografias podem ser úteis também em SAR, mas a diferença de domínio exige avaliação; não se pressupõe que a transferência melhore os resultados.

As três estratégias estão implementadas e são selecionadas uma por rodada; sua execução completa e comparação dependem de treinamento real:

| `experiment` | Treinável | Congelado |
| --- | --- | --- |
| `frozen_backbone` | RPN e cabeças de caixa e máscara | ResNet-50 e FPN |
| `partial_finetune` | FPN, RPN, cabeças e blocos finais do ResNet-50 | Blocos iniciais do ResNet-50 |
| `full_finetune` | Todo o modelo | Nada |

A estratégia parcial usa `trainable_layers=2` por padrão, liberando `layer3` e `layer4`; o construtor oficial recebe 0, o valor configurado ou 5, respectivamente. Os blocos congelados ficam em `eval()` durante o treino para impedir a atualização das estatísticas de BatchNorm. Manter os demais parâmetros e divisão iguais ajuda a comparar as estratégias, mas não há resultado de desempenho afirmado aqui.

## Desenho experimental

### Pergunta e variáveis

A pergunta é **até que profundidade o Mask R-CNN treinado no COCO precisa ser readaptado para detectar e segmentar navios em SAR**. Diferente do DeepPCB, aqui não há rodada sem pré-treino: todas as estratégias partem dos mesmos pesos COCO e mudam apenas a **fronteira entre o que fica congelado e o que treina**. O estudo, portanto, não mede "transferência × treino do zero". Ele mede **quanto da rede precisa ser adaptado** ao novo domínio.

- **Variável independente:** `experiment` e, dentro de `partial_finetune`, `trainable_layers`.
- **Variáveis controladas:** pesos iniciais, divisão por cena, semente, AdamW com a mesma taxa e o mesmo `weight_decay`, `CosineAnnealingLR`, `epochs`, `patience`, `batch_size`, espelhamento horizontal e limiares de avaliação.
- **Variáveis de resposta:** mAP de caixa e de máscara (`@0.50` e `@0.50:0.95`), precisão, recall e F1 de caixa e de máscara, contagens TP/FP/FN e matriz de confusão, componentes da perda de treino, época do melhor checkpoint e número de parâmetros treináveis (`reports/parameters.csv`).

### Por que o domínio importa

Imagens SAR não são fotografias. O sensor mede o retroespalhamento do sinal de radar emitido pelo próprio satélite, e não a luz refletida. Isso traz diferenças que afetam cada nível da rede:

- **Um canal só.** A imagem é monocromática e é replicada em três canais RGB iguais para caber na entrada do modelo. Os filtros do COCO que respondem a contrastes de cor recebem a mesma informação nos três canais.
- **Ruído *speckle*.** É um ruído granular e multiplicativo, próprio da formação coerente da imagem. As texturas finas que as primeiras camadas detectam são, em boa parte, esse ruído, e não detalhe do objeto.
- **Estatística diferente.** Mar escuro e homogêneo, navios como alvos pequenos e muito brilhantes, e costa com estruturas brilhantes que se parecem com navios. A média e a variância das ativações em cada camada diferem das do COCO. Isso afeta diretamente o BatchNorm congelado.
- **Escala e ponto de vista.** Vista de cima, sem perspectiva, com a maioria dos alvos ocupando poucos pixels. No COCO, objetos pequenos são minoria e quase tudo é visto de lado.

A hipótese de trabalho, que cada estratégia testa em um grau diferente, é a habitual em transfer learning: **camadas rasas transferem melhor que camadas profundas**. Bordas e manchas brilhantes são úteis em qualquer imagem, enquanto as combinações de alto nível do COCO (rodas, rostos, pernas) dificilmente servem para navios em radar.

### Estratégia `frozen_backbone`

**O que é.** `trainable_backbone_layers=0` congela os cinco estágios do ResNet-50, e o notebook congela também a FPN. Ambos ficam em `eval()`, preservando as estatísticas de BatchNorm do COCO. Treinam a RPN, `box_head`, `mask_head` (todas partindo do COCO) e os dois preditores novos de duas classes.

**O que mede.** Se a **pirâmide de características do COCO, intacta**, já contém informação suficiente para localizar navios e delinear sua forma. É o teste mais direto da hipótese de que a representação transfere sem ajuste.

**Mecanismo e expectativas teóricas.** Como a FPN também está congelada, todas as escalas da pirâmide, inclusive os níveis de alta resolução usados para alvos pequenos, ficam como foram aprendidas no COCO. Se as ativações para um navio pequeno sobre o mar não se destacam nesses níveis, a RPN não tem como propor a região, e o erro aparece como **FN**. A cabeça de máscara depende do mesmo mapa recortado pelo RoI Align. Se a característica é pobre em bordas precisas, a máscara pode ficar pior que a caixa, o que aparece como diferença entre a mAP de máscara e a de caixa. A vantagem é que há poucos parâmetros treináveis, então o risco de sobreajuste é o menor das três.

**Sinais para observar.** Convergência rápida seguida de platô; recall baixo, com FN dominando a matriz de confusão, principalmente em navios pequenos; `loss_objectness` e `loss_rpn_box_reg` que param de cair cedo, indicando que a RPN não consegue separar navio de mar com o mapa fixo.

**Custo.** É o menor: não há gradiente no backbone nem na FPN.

### Estratégia `partial_finetune`

**O que é.** Os blocos finais do ResNet-50 (`trainable_layers`, padrão `2`: `layer3` e `layer4`), a FPN inteira, a RPN e as cabeças treinam. Os blocos iniciais (`conv1`, `bn1`, `layer1`, `layer2` no padrão) ficam congelados e em `eval()`.

**O que mede.** Se basta **reaproveitar as características de baixo nível do COCO e readaptar as de alto nível** ao SAR. É o meio-termo entre estabilidade e adaptação, e testa diretamente a hipótese de que camadas rasas transferem melhor.

**Mecanismo e expectativas teóricas.** As camadas rasas, que detectam bordas e manchas, continuam como no COCO e recebem estatísticas de BatchNorm do COCO. As camadas profundas e a FPN podem aprender a combinar essas bordas em "alvo brilhante compacto sobre fundo escuro". A FPN treinável é importante: ela pode reponderar os níveis da pirâmide para os alvos pequenos do HRSID. Comparado ao fine-tuning completo, há menos parâmetros livres e os primeiros filtros não são perturbados por gradientes ruidosos no início do treino.

**`trainable_layers` como sub-experimento.** O valor define a fronteira de congelamento e pode ser varrido de `1` a `4`:

| `trainable_layers` | Blocos do ResNet-50 que treinam | Blocos congelados |
| --- | --- | --- |
| `1` | `layer4` | `conv1`, `bn1`, `layer1`–`layer3` |
| `2` (padrão) | `layer3`, `layer4` | `conv1`, `bn1`, `layer1`, `layer2` |
| `3` | `layer2`–`layer4` | `conv1`, `bn1`, `layer1` |
| `4` | `layer1`–`layer4` | `conv1`, `bn1` |

Varrer esses valores, com tudo o mais fixo, mostra **em que profundidade a transferência deixa de ajudar**: o ponto em que liberar mais um bloco já não melhora a validação ou começa a sobreajustar. Junto com `frozen_backbone` (equivalente a `0`, mas com a FPN também congelada) e `full_finetune` (`5`), forma uma curva completa da profundidade de adaptação.

**Sinais para observar.** Compare com `frozen_backbone` em recall e FN, para ver se adaptar o topo da rede recupera alvos perdidos. Compare com `full_finetune` na distância entre perda de treino e mAP de validação, para ver se congelar a base reduz o sobreajuste.

**Custo.** É intermediário e cresce com `trainable_layers`.

### Estratégia `full_finetune`

**O que é.** `trainable_backbone_layers=5`. Todos os parâmetros treinam, e nenhum módulo é forçado a `eval()`: todo BatchNorm passa a acumular estatísticas das imagens SAR.

**O que mede.** Se a adaptação **também das primeiras camadas** compensa. Por exemplo, filtros iniciais que aprendam a lidar com o *speckle* em vez de responder a ele como textura.

**Mecanismo e expectativas teóricas.** É a estratégia com maior capacidade de adaptação ao domínio e, por isso, com maior risco de sobreajuste e de esquecer o que o COCO ensinou. Dois detalhes do notebook pesam aqui:

- **BatchNorm com batch pequeno.** O V2 usa `BatchNorm2d` comum. Com `batch_size=2`, as médias e variâncias de cada passo do backbone são estimadas com duas imagens, e as médias móveis podem ficar ruidosas. Esse efeito não existe nas camadas congeladas das outras estratégias. Ele é uma diferença real entre os tratamentos, e não só "mais parâmetros".
- **Mesma taxa para todas as camadas.** O AdamW aplica `learning_rate` igualmente a backbone e cabeças. Em fine-tuning completo, é comum usar taxas menores nas camadas rasas (*discriminative learning rates*), o que o notebook não faz.

**Sinais para observar.** Perda de treino bem abaixo das outras estratégias, com mAP de validação estagnada ou caindo (sobreajuste); oscilações na validação de uma época para outra (BatchNorm ruidoso ou taxa alta); melhor checkpoint em época precoce.

**Custo.** É o maior: gradientes e estados do otimizador para todos os parâmetros. Com AdamW, os estados são dois tensores por parâmetro.

### O que cada comparação isola

| Comparação | O que muda | O que a comparação responde |
| --- | --- | --- |
| `partial_finetune` × `frozen_backbone` | Libera FPN e blocos finais do ResNet-50 | O ganho de adaptar as características de alto nível ao SAR |
| `full_finetune` × `partial_finetune` | Libera os blocos iniciais e seus BatchNorm | Se as características de baixo nível do COCO também precisam mudar |
| `partial_finetune` com `trainable_layers` de 1 a 4 | Uma camada por vez | A profundidade ideal da fronteira de congelamento |
| Mask mAP − Box mAP, em cada estratégia | Mesmo modelo, outra métrica | O custo de segmentar além de localizar, e se ele muda com a adaptação |

A diferença entre a mAP de máscara e a de caixa merece leitura própria. Ela separa dois tipos de erro: **achar o navio** (RPN e cabeça de caixa) e **delinear o navio** (cabeça de máscara sobre o RoI Align). Uma estratégia pode melhorar uma coisa sem melhorar a outra.

### Ameaças à validade

- **Hiperparâmetros compartilhados.** Taxa de aprendizado, `weight_decay` e número de épocas são os mesmos para as três estratégias, embora o valor ideal de cada um dependa de quantos parâmetros treinam. A comparação vale **sob esta configuração comum**. Ajustar a taxa por estratégia, sempre pela validação, é uma extensão possível.
- **Orçamento de épocas e early stopping.** `frozen_backbone` tende a convergir antes e pode parecer melhor com poucas épocas. Registre a época do melhor checkpoint: se ela coincidir com a última, a rodada não terminou de aprender.
- **Uma semente.** Diferenças pequenas entre estratégias podem estar dentro do ruído entre sementes. Repita cada estratégia com algumas sementes e relate média e dispersão.
- **Não há linha de base sem pré-treino.** Sem uma rodada partindo de pesos aleatórios, o estudo não quantifica o ganho do COCO em si, apenas o de adaptar mais ou menos camadas. Para isso, seria preciso acrescentar uma estratégia com `weights=None`, como a A do DeepPCB.
- **Divisão e vazamento.** A seleção usa a validação separada por cena. O teste oficial compartilha cenas com o treino e deve ser avaliado uma única vez, no final (veja *Divisão e vazamento*).
- **Mistura de regimes e tamanhos.** Uma única mAP mistura navios em mar aberto (fáceis de separar do fundo) e junto à costa (confundidos com estruturas), e também mistura alvos pequenos e grandes. Uma estratégia pode ganhar em um grupo e perder em outro sem que o número agregado mostre isso.

### Protocolo sugerido

1. Fixe `CONFIG`, exceto `experiment` e `trainable_layers`, e não altere mais nada entre as rodadas.
2. Rode `frozen_backbone`, `partial_finetune` e `full_finetune` com `run_training=True` e `evaluate_test=False`. Opcionalmente, varra `trainable_layers` em `partial_finetune`.
3. Compare na validação: `history.csv`, curvas, `mask_map_50_95` e `box_map_50_95` do melhor checkpoint, época em que ele ocorreu, componentes da perda, matriz de confusão e parâmetros treináveis.
4. Repita com outras sementes antes de tirar conclusões.
5. Escolha a estratégia pela validação e só então avalie o teste, uma vez, com `evaluate_test=True` e `checkpoint_path`.

## Configuração e treinamento

Valores atuais em `CONFIG`:

| Parâmetro | Padrão | Função |
| --- | --- | --- |
| `seed` | `42` | Sementes de Python, NumPy e PyTorch |
| `experiment`, `trainable_layers` | `frozen_backbone`, `2` | Estratégia e blocos finais no fine-tuning parcial |
| `epochs`, `patience` | `15`, `5` | Máximo de épocas e parada sem melhora |
| `batch_size`, `num_workers` | `2`, `2` | Imagens por batch e processos de leitura |
| `amp`, `device` | `True`, `auto` | AMP apenas em CUDA; seleção automática ou `cuda`, `mps`, `cpu` |
| `learning_rate`, `weight_decay` | `0.0001`, `0.0005` | AdamW nos parâmetros treináveis |
| `val_fraction`, `horizontal_flip` | `0.15`, `0.5` | Fração aproximada de validação e probabilidade de espelhamento |
| `confidence_threshold`, `iou_threshold`, `mask_threshold` | `0.5`, `0.5`, `0.5` | Limiares de score, correspondência e binarização de máscaras |
| `dataset_dir`, `results_dir` | `''`, `None` | Caminhos dos dados extraídos e dos resultados |
| `limit_images`, `sample_images` | `None`, `200` | Limite opcional por split e amostra de checagem de pixels |
| `run_training`, `evaluate_test`, `checkpoint_path` | `True`, `False`, `None` | Treinamento, teste e checkpoint externo |

Após alterar `CONFIG`, reexecute sua célula e recrie os DataLoaders se mudar seus parâmetros. O `collate_fn` preserva listas de imagens e alvos de tamanhos variáveis; somente treino embaralha e aumenta dados. `CosineAnnealingLR` agenda a taxa ao longo das épocas. A perda de treino soma as perdas de classificação, regressão de caixa, máscara, classificação de objetos e regressão da RPN; componentes e total são registrados. Perda não finita interrompe o treino.

Os padrões do TorchVision incluem redimensionamento mínimo/máximo de 800/1333, âncoras 32–512 por nível, score mínimo interno 0,05, NMS de caixa 0,5 e até 100 detecções por imagem; podem variar com a versão instalada. O limite de 100 pode restringir imagens com muitos navios. O cálculo de máscaras e mAP pode exigir memória e tempo consideráveis; `batch_size`, `limit_images`, dispositivo e acesso ao Drive afetam o custo. AMP não altera a resolução ou a arquitetura.

## Avaliação e resultados

A validação ocorre a cada época. TorchMetrics com backend pycocotools calcula separadamente `box_map_50`, `box_map_50_95`, `mask_map_50` e `mask_map_50_95`. Precisão, recall e F1 de caixa e máscara usam correspondência gulosa por score e os limiares de `CONFIG`, com TP, FP e FN. A mAP não usa o filtro adicional de confiança, mas o detector aplica seus próprios filtros; a mAP de máscara depende de `mask_threshold`, usado para binarizar as máscaras. Não há cálculo de perda de validação.

O melhor checkpoint é escolhido pelo maior `mask_map_50_95` na validação; `patience` controla a parada antecipada. O teste oficial é opcional (`evaluate_test=True`) e deve ficar reservado até a escolha da estratégia. Sem treino na mesma execução, forneça `checkpoint_path`, relativo à raiz do projeto ou absoluto. Carregue apenas checkpoints confiáveis, pois `torch.load` usa `weights_only=False`. O checkpoint salva estados de otimizador, scheduler e scaler, mas não há rotina de retomada de treino.

No destino padrão `runs/results/hrsid_mask_rcnn_v2/` do ambiente ativo, `reports/` e `figures/` na raiz contêm a análise do dataset. Cada rodada em `<experimento>/<data_hora>/` guarda `config.json`, divisão, contagem de parâmetros, histórico, métricas de validação e, quando solicitadas, de teste em `reports/`; `checkpoints/best.pt`; curvas individuais e painel em `figures/`; e visualizações em `predictions/`. As figuras de predição comparam imagem, anotação e saída, priorizando imagens de validação com mais navios.

A matriz de confusão, mostrada após as predições, é 2 × 2 (navio × fundo) para caixas e para máscaras. Ela é montada a partir das contagens TP, FP e FN já salvas: de `reports/metrics_test.json`, se o teste foi avaliado, ou das métricas de validação da melhor época guardadas no checkpoint. Por isso, aparece sem treinar nem reavaliar o modelo. O canto fundo × fundo (verdadeiro negativo) não é definido em detecção, porque não existe um conjunto finito de "regiões sem navio" para contar, e fica em branco. A figura é salva como `figures/confusion_<split>.png` na pasta da rodada. A localização depende de como o checkpoint é informado: ao apenas configurar `checkpoint_path` sem treino nem teste, usa-se o diretório geral de predições; quando há treino ou avaliação de teste, o caminho é derivado do checkpoint.

Sementes facilitam a repetição, sem garantir determinismo em toda GPU. Avaliação por tamanho e por regime costeiro/oceânico e repetição com outras sementes são análises futuras; o notebook atual não registra mAP por tamanho ou regime. Os artefatos de cada execução são a fonte para resultados observados, e nenhum desempenho é presumido aqui.

[Índice](README.md).
