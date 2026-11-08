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

No destino padrão `runs/results/hrsid_mask_rcnn_v2/` do ambiente ativo, `reports/` e `figures/` na raiz contêm a análise do dataset. Cada rodada em `<experimento>/<data_hora>/` guarda `config.json`, divisão, contagem de parâmetros, histórico, métricas de validação e, quando solicitadas, de teste em `reports/`; `checkpoints/best.pt`; curvas individuais e painel em `figures/`; e visualizações em `predictions/`. As figuras de predição comparam imagem, anotação e saída, priorizando imagens de validação com mais navios. A localização depende de como o checkpoint é informado: ao apenas configurar `checkpoint_path` sem treino nem teste, usa-se o diretório geral de predições; quando há treino ou avaliação de teste, o caminho é derivado do checkpoint.

Sementes facilitam a repetição, sem garantir determinismo em toda GPU. Avaliação por tamanho e por regime costeiro/oceânico e repetição com outras sementes são análises futuras; o notebook atual não registra mAP por tamanho ou regime. Os artefatos de cada execução são a fonte para resultados observados, e nenhum desempenho é presumido aqui.

[Índice](README.md).
