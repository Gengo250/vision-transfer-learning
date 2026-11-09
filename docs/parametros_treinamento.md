# DeepPCB: detecção de defeitos e transfer learning

Este documento descreve [deep_pcb_transfer_learning.ipynb](../notebooks/deep_pcb_transfer_learning.ipynb). O objetivo é comparar três inicializações e políticas de treinamento do mesmo detector. Consulte o [ambiente compartilhado](environment.md) para instalação e execução.

## Dados e integridade

O [DeepPCB oficial](https://github.com/tangsanli5201/DeepPCB) reúne 1.500 pares de imagens de placas, com uma imagem inspecionada e um template alinhado sem defeitos. O dataset usado anteriormente neste projeto tinha 10.013 anotações; esses totais dependem da cópia e não foram recontados nesta revisão. A resolução informada é 640 × 640 pixels. Uma cópia foi disponibilizada no [Google Drive](https://drive.google.com/drive/folders/1mdTe8Ez5r7KCi5ryaBf_onUCZJHOlPqH?usp=drive_link).

A pasta esperada contém `trainval.txt`, `test.txt` e diretórios de grupos. Cada linha dos splits aponta para uma imagem-base e seu arquivo de anotações; os arquivos de imagem correspondentes terminam em `_test.jpg` e `_temp.jpg`. Cada linha da anotação contém `x1 y1 x2 y2 classe`, com caixas XYXY em pixels absolutos. Os IDs de 1 a 6 representam, respectivamente, `open`, `short`, `mousebite`, `spur`, `copper` e `pin-hole`; 0 é fundo.

O padrão é `data/DeepPCB/PCBData/` localmente e `MyDrive/vision-transfer-learning/dataset/DeepPCB/PCBData/` no Colab. `CONFIG['dataset_dir']` é relativo a `data/` ou `dataset/`, respectivamente; caminhos absolutos também podem ser usados. O notebook não baixa nem copia os dados.

O leitor percorre os arquivos listados, conta imagens e anotações, mostra a distribuição por classe e os grupos compartilhados; também exibe uma imagem inspecionada com caixas ao lado do template. A ausência da pasta gera erro e arquivos ilegíveis ou anotações malformadas podem interromper a leitura. **Não há auditoria completa automatizada** de dimensões, duplicatas, limites das caixas, existência de todos os templates ou integridade dos pixels. Confira a origem e a integridade da cópia antes do treino.

## Divisão e preparação

`trainval.txt` é dividido por grupos inteiros, buscando cobrir as seis classes nos dois lados e aproximar 15% das imagens na validação; a semente desempata candidatos. `test.txt` permanece como teste oficial. Na cópia anteriormente auditada, a divisão foi 850/150/500 imagens para treino/validação/teste. As contagens reais são impressas ao executar o notebook.

Treino e validação não compartilham grupos, mas o teste oficial compartilha grupos com o desenvolvimento. Assim, o teste não demonstra generalização para placas ou grupos independentes. O template é apenas visualizado: o detector recebe **somente** a imagem `_test.jpg`, convertida para RGB e tensor `float32` em `[0, 1]`. Não há data augmentation implementado. O modelo faz normalização interna e redimensiona, por padrão, com `min_size=800`, `max_size=1333`; a ampliação não cria novos detalhes. Testar outra resolução é uma proposta, mantendo-a igual em A/B/C para comparação.

## Modelo e estratégias

O modelo é o [`fasterrcnn_resnet50_fpn_v2`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html), não um Mask R-CNN. O ResNet-50 extrai características, a FPN fornece escalas múltiplas, a RPN propõe regiões e as cabeças RoI classificam e refinam caixas. Em inferência, cada imagem produz caixas XYXY, classes e scores; não há máscaras. Os alvos incluem caixas, rótulos, `image_id`, área e `iscrowd`.

`FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT` corresponde a pesos COCO; a cabeça `FastRCNNPredictor` é substituída por uma com sete classes, incluindo fundo. O notebook implementa uma estratégia por rodada:

| `experiment` | Inicialização | Parâmetros treinados | BatchNorm do backbone no treino |
| --- | --- | --- | --- |
| `A` | Aleatória, sem COCO nem ImageNet | Todo o detector | Atualiza estatísticas |
| `B` | COCO, com preditor novo | RPN e cabeças; ResNet-50 e FPN congelados | Fixo em `eval()` com estatísticas COCO |
| `C` | COCO, com preditor novo | Todo o detector | Atualiza estatísticas |

## Desenho experimental

### Pergunta e variáveis

A pergunta do estudo é **quanto o conhecimento aprendido em fotografias naturais (COCO) ajuda a detectar defeitos em placas de circuito**, e em que parte da rede esse conhecimento precisa ser ajustado. Para isso, uma única variável muda entre as rodadas: a estratégia em `CONFIG['experiment']`. Ela define a inicialização dos pesos e quais parâmetros recebem gradiente.

- **Variável independente:** a estratégia A, B ou C.
- **Variáveis controladas:** arquitetura, divisão por grupos, semente, otimizador SGD e sua taxa de aprendizado, StepLR, número máximo de épocas, `patience`, `batch_size`, resolução de entrada e limiares de avaliação. Elas só podem ser comparadas se forem iguais nas três rodadas.
- **Variáveis de resposta:** `map_50_95` e `map_50` de validação, precisão, recall e F1 no ponto de operação de `CONFIG`, matriz de confusão, perda de treino por época e época do melhor checkpoint.

### Por que o domínio importa

O COCO contém fotografias coloridas de objetos do cotidiano, com fundos variados e objetos ocupando boa parte da imagem. O DeepPCB é quase o oposto: imagens em tons de cinza, de alto contraste, com padrões geométricos repetitivos (trilhas, pads e furos) e defeitos que ocupam poucos pixels. Além disso, um defeito costuma ser **uma pequena alteração de um padrão regular**, e não um objeto com forma própria. Um `spur` ou um `mousebite` é uma irregularidade na borda de uma trilha.

Isso cria uma tensão que as três estratégias exploram:

- As primeiras camadas de uma CNN aprendem filtros de borda, cantos e texturas simples, que costumam ser úteis em qualquer domínio. Esse é o argumento a favor da transferência.
- As camadas profundas do ResNet-50 e a FPN foram ajustadas para combinar essas bordas em partes de pessoas, carros e animais. Essas combinações podem não representar bem "trilha com um pedaço faltando". Esse é o argumento a favor de adaptar o backbone.

### Estratégia A: treino do zero

**O que é.** O detector inteiro, incluindo ResNet-50, FPN, RPN e cabeças, começa com pesos aleatórios. `weights=None` e `weights_backbone=None` impedem que o TorchVision carregue COCO ou ImageNet. Todos os parâmetros treinam e o BatchNorm aprende estatísticas apenas das imagens de PCB.

**O que mede.** É a **linha de base sem transferência**. Sem A, não há como afirmar que B ou C "ganharam" com o pré-treino: qualquer número de B ou C só vira evidência de transferência quando comparado a A sob o mesmo orçamento.

**Mecanismo e expectativas teóricas.** Um ResNet-50 com FPN tem dezenas de milhões de parâmetros, e o conjunto de treino tem algumas centenas de imagens. Redes profundas treinadas do zero com poucos dados tendem a convergir devagar e a depender mais de regularização e de muitas épocas. Por outro lado, o domínio é visualmente simples e repetitivo, o que pode reduzir essa desvantagem. Em [*Rethinking ImageNet Pre-training*](https://arxiv.org/abs/1811.08883), He et al. mostram que detectores treinados do zero podem alcançar os pré-treinados no próprio COCO, mas com muito mais iterações. Os mesmos autores observam que, com poucos dados, a desvantagem do treino do zero aumenta. Aqui, além disso, o orçamento é de no máximo `epochs=20`, com a mesma taxa de aprendizado das outras estratégias. Portanto, A responde "quanto se consegue **neste orçamento**", e não "qual o teto do treino do zero".

**Sinais para observar nas curvas.** Perda de treino que cai lentamente nas primeiras épocas; mAP de validação próxima de zero no início; instabilidade ou perda não finita, caso a taxa de aprendizado seja alta para pesos aleatórios. Se o early stopping disparar muito cedo, a rodada pode estar subestimando A.

**Custo.** É igual ao de C por passo, porque todos os gradientes são calculados.

### Estratégia B: extrator de características congelado

**O que é.** O modelo parte dos pesos COCO. ResNet-50 e FPN ficam com `requires_grad=False` e em `eval()` durante o treino. Assim, nem os pesos nem as médias e variâncias do BatchNorm mudam. Só a RPN e as cabeças RoI (`box_head` e o preditor novo de sete classes) aprendem.

**O que mede.** Se as **características do COCO, como estão**, já separam os defeitos do fundo e entre si. A RPN e as cabeças precisam aprender a usar um mapa de características fixo.

**Mecanismo e expectativas teóricas.** Congelar reduz muito o número de parâmetros treináveis, o que diminui o risco de sobreajuste com poucos dados e acelera cada passo, porque não há retropropagação no backbone. O limite é a representação: se a FPN não produz ativações distintas para um `spur` e uma trilha normal, as cabeças não têm como recuperar essa informação. B tende a ir bem quando o domínio de origem e o de destino compartilham padrões visuais, e a ficar limitado quando o domínio é muito diferente. Manter o BatchNorm em `eval()` é essencial: com ele em modo de treino, as estatísticas mudariam para as de PCB mesmo sem gradiente, e o "congelamento" deixaria de ser real.

**Sinais para observar nas curvas.** Convergência rápida seguida de um platô, porque a capacidade de adaptação é pequena. Uma diferença grande entre `map_50` e `map_50_95` indica que o modelo encontra o defeito, mas localiza a caixa com pouca precisão. A matriz de confusão mostra se as classes parecidas (`spur` × `mousebite`, `open` × `short`) se confundem quando o backbone não é ajustado.

**Custo.** É o menor das três: menos memória para gradientes e passos mais curtos.

### Estratégia C: fine-tuning completo

**O que é.** O modelo parte dos pesos COCO e todos os parâmetros treinam, incluindo o BatchNorm, que passa a acumular estatísticas das imagens de PCB.

**O que mede.** Se partir do COCO e **adaptar a rede inteira** ao domínio supera tanto não ter pré-treino (A) quanto não adaptar o backbone (B).

**Mecanismo e expectativas teóricas.** C combina uma boa inicialização com liberdade total de adaptação. Os riscos são dois. O primeiro é sobreajustar o conjunto pequeno de treino, porque há tantos parâmetros livres quanto em A. O segundo é apagar rapidamente características úteis do COCO se a taxa de aprendizado for alta (*catastrophic forgetting*). Há ainda um detalhe do V2: ele usa `BatchNorm2d` comum, e com `batch_size=2` as estatísticas de cada passo são estimadas com apenas duas imagens. Isso pode deixar as médias móveis ruidosas, algo que não acontece em B.

**Sinais para observar nas curvas.** Perda de treino caindo bem abaixo da de B, enquanto a mAP de validação para de subir ou cai, indica sobreajuste. Nesse caso, `patience` e a escolha do melhor checkpoint protegem a rodada.

**Custo.** É o maior junto com A.

### O que cada comparação isola

| Comparação | Diferença entre as rodadas | O que a comparação responde |
| --- | --- | --- |
| C × A | Só a inicialização (COCO × aleatória); ambas treinam tudo | O valor do pré-treino em si |
| C × B | Só o que treina (tudo × apenas RPN e cabeças); ambas partem do COCO | O valor de adaptar o backbone ao domínio |
| B × A | Inicialização e parâmetros treinados mudam juntos | Se características COCO congeladas superam aprender tudo do zero; não separa os dois efeitos |

### Ameaças à validade

- **Hiperparâmetros compartilhados.** A mesma taxa de aprendizado e o mesmo agendamento servem às três estratégias, mas o melhor valor para cada uma tende a ser diferente: treino do zero e fine-tuning costumam preferir taxas distintas. A comparação mede as estratégias **sob uma configuração comum**, e não cada uma no seu ótimo. Ajustar a taxa por estratégia, usando apenas a validação, é uma extensão possível.
- **Orçamento de épocas.** Um mesmo máximo de épocas favorece estratégias que convergem rápido. Registre a época do melhor checkpoint: se ela coincidir com a última, a rodada ainda estava melhorando.
- **Uma semente.** Diferenças pequenas entre estratégias podem estar dentro da variação entre sementes. Repetir cada estratégia com algumas sementes e relatar média e dispersão torna a conclusão defensável.
- **Mesma divisão para tudo.** A escolha da estratégia deve usar apenas a validação. O teste oficial é avaliado uma vez, no final, e compartilha grupos com o desenvolvimento (veja a seção de divisão).
- **Template ignorado.** Na inspeção industrial, o template sem defeitos é uma informação muito forte, e métodos baseados em diferença entre imagens a exploram. Este estudo usa só a imagem inspecionada para medir a transferência de um detector genérico. Por isso, os números não representam o que se alcança com o par imagem-template.

### Protocolo sugerido

1. Fixe `CONFIG` (exceto `experiment`) e não altere mais nada entre as rodadas.
2. Rode A, B e C com `run_training=True` e `evaluate_test=False`; cada rodada cria sua pasta em `runs/results/`.
3. Compare as rodadas na validação: `history.csv`, curvas, `map_50_95` do melhor checkpoint, época em que ele ocorreu e matriz de confusão.
4. Se possível, repita com outras sementes antes de tirar conclusões.
5. Escolha a estratégia final pela validação e só então avalie o teste, uma única vez, com `evaluate_test=True` e `checkpoint_path`.

## Configuração e treinamento

Valores atuais da célula `CONFIG`:

| Parâmetro | Padrão | Função |
| --- | --- | --- |
| `seed` | `42` | Sementes Python, NumPy e PyTorch e desempate dos grupos |
| `dataset_dir` | `DeepPCB/PCBData` | Pasta dos dados relativa à raiz de dados do ambiente |
| `experiment` | `C` | Estratégia A, B ou C |
| `epochs`, `patience` | `20`, `5` | Máximo de épocas e parada após épocas sem melhora |
| `batch_size`, `num_workers` | `2`, `0` | Imagens por batch e processos de leitura |
| `amp` | `True` | Precisão mista com GradScaler apenas em CUDA |
| `learning_rate`, `momentum`, `weight_decay` | `0.001`, `0.9`, `0.0005` | SGD nos parâmetros treináveis |
| `lr_step_size`, `lr_gamma` | `7`, `0.1` | StepLR reduz a taxa por fator 0,1 a cada sete épocas |
| `score_threshold`, `iou_threshold` | `0.5`, `0.5` | Limiares das métricas pontuais e matriz de confusão |
| `run_training`, `evaluate_test` | `True`, `False` | Iniciar treino e, opcionalmente, avaliar o teste |
| `checkpoint_path` | `None` | Checkpoint para teste sem treino na mesma execução |

Após mudar `CONFIG`, reexecute sua célula; para `batch_size` ou `num_workers`, recrie os DataLoaders. Um `collate_fn` permite números distintos de alvos por imagem e somente o treino embaralha os dados. A perda de treino é a soma das perdas retornadas pelo TorchVision: classificação e regressão de caixas da RPN e das cabeças de detecção. Uma perda não finita interrompe a rodada. O checkpoint guarda também otimizador, scheduler e scaler, mas o notebook não implementa retomada de treino.

O registro anterior de parâmetros relatava, para uma RTX 3050 Laptop de 4 GB com batch 2 e imagens a 800 px, passos de aproximadamente 725 ms em FP32 e 364 ms com AMP, com picos de 2,7 e 1,7 GB, respectivamente. Também relatava leitura de imagem em cerca de 3,5 ms. Esses números históricos não têm protocolo ou artefato de medição associado nesta documentação e não foram verificados nesta revisão; não devem ser generalizados para outras máquinas ou rodadas.

Os padrões do detector que não são configurados no notebook incluem âncoras de 32 a 512 pixels por nível, normalização com médias e desvios ImageNet, `box_score_thresh=0.05`, `box_nms_thresh=0.5` e no máximo 100 detecções por imagem. Esses padrões dependem da versão do TorchVision; o limite pode afetar imagens com muitos defeitos.

## Avaliação e artefatos

A cada época, a validação calcula mAP de caixas no protocolo COCO via TorchMetrics/pycocotools: `map_50` e `map_50_95`. A mAP usa os scores retornados pelo detector, sem o filtro adicional `score_threshold`; o detector ainda aplica seus próprios filtros e NMS. O benchmark DeepPCB original usa IoU 0,33, portanto seus números não são diretamente comparáveis aos deste protocolo.

Precisão, recall e F1 são micro, usando confiança e IoU de `CONFIG` e casamento guloso por score que exige a classe correta. A matriz de confusão casa caixas por posição independentemente da classe, com linhas reais e colunas previstas, incluindo fundo. O melhor checkpoint é escolhido pelo maior `map_50_95` de validação; o early stopping monitora a mesma métrica. Não há perda de validação calculada.

Para reservar o teste à avaliação final, escolha a estratégia com validação e só depois use `evaluate_test=True`. Com `run_training=False`, informe `checkpoint_path`; com treino na mesma execução, usa-se o melhor checkpoint criado. Caminhos relativos partem da raiz do projeto. Carregue apenas checkpoints confiáveis: `torch.load` usa `weights_only=False`.

Cada rodada cria `runs/results/<data_hora>_<experimento>/` no ambiente ativo com `config.json`, `split.csv`, `history.csv`, `checkpoints/best.pt`, curvas individuais e painel em `figures/`, matriz de validação em `.png` e `.npy` e, se solicitado, `test_metrics.json` e matriz de teste. A visualização implementada mostra exemplos anotados e matrizes; não há rotina de visualização de predições sobre a imagem. Sementes ajudam a reproduzir divisões e inicialização, mas não garantem determinismo em todos os dispositivos.

Os resultados dependem do dataset e da execução; nenhuma métrica de desempenho de treinamento completo é afirmada nesta documentação.

[Índice](README.md).
