import numpy as np
from .morph_core import indices_delaunay  # os alunos podem reutilizar

# ------------------------- Funções a implementar pelos estudantes -------------------------

def pontos_medios(pA, pB):
    """
    Retorna os pontos médios (N,2) entre pA e pB.
    """
    return (pA + pB) * 0.5

def indices_pontos_medios(pA, pB):
    """
    Calcula a triangulação de Delaunay nos pontos médios e retorna (M,3) int.
    Dica: use pontos_medios + indices_delaunay().
    """
    pM = pontos_medios(pA, pB)
    return indices_delaunay(pM)

# Interpoladoras
def linear(t, a=1.0, b=0.0):
    """
    Interpolação linear: a*t + b (espera-se mapear t em [0,1]).
    """
    return a*t + b

def sigmoide(t, k):
    """
    Sigmoide centrada em 0.5, normalizada para [0,1].
    k controla a "inclinação": maior k => transição mais rápida no meio.
    """
    x = k * (t - 0.5)
    return 1.0 / (1.0 + np.exp(-x))

def dummy(t):
    """
    Função 'dummy' que pode ser usada como exemplo de função constante.
    """
    return 0.5

# Geometria / warping por triângulos
def _det3(a, b, c):
    """
    Determinante 2D para área assinada (auxiliar das baricêntricas).
    """
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - \
           (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

def _transf_baricentrica(pt, tri):
    """
    pt: (x,y)
    tri: (3,2) com vértices v1,v2,v3
    Retorna (w1,w2,w3); espera-se w1+w2+w3=1 quando pt está no plano do tri.
    """
    # Extrai os vértices
    v1, v2, v3 = tri[0], tri[1], tri[2]

    # Calcula a área (x2) do triângulo principal (v1, v2, v3)
    det_total = _det3(v1, v2, v3)

    # Lida com triângulos degenerados (área zero)
    if np.isclose(det_total, 0.0):
        return 1.0, 0.0, 0.0

    # w1 é a razão da área do triângulo (pt, v2, v3) pela área total
    det_w1 = _det3(pt, v2, v3)
    
    # w2 é a razão da área do triângulo (v1, pt, v3) pela área total
    det_w2 = _det3(v1, pt, v3)

    # Calcula os pesos
    w1 = det_w1 / det_total
    w2 = det_w2 / det_total
    
    # calculado usando a propriedade de que a soma deve ser 1.
    w3 = 1.0 - w1 - w2

    return w1, w2, w3

def _check_bari(w1, w2, w3, eps=1e-6):
    """
    Testa inclusão de ponto no triângulo usando baricêntricas (com tolerância).
    """
    return (w1 >= -eps) and (w2 >= -eps) and (w3 >= -eps)

def _tri_bbox(tri, W, H):
    """
    Retorna bounding box inteiro (xmin,xmax,ymin,ymax), recortado ao domínio [0..W-1],[0..H-1].
    """
    # Encontra os valores flutuantes mínimos e máximos (x, y) do triângulo
    # min_coords será [xmin_f, ymin_f]
    # max_coords será [xmax_f, ymax_f]
    min_coords = np.amin(tri, axis=0)
    max_coords = np.amax(tri, axis=0)

    # Converte para coordenadas inteiras
    # Usamos floor() para o mínimo e ceil() para o máximo para garantir
    # que o bounding box contenha totalmente o triângulo.
    xmin_i = np.floor(min_coords[0]).astype(int)
    ymin_i = np.floor(min_coords[1]).astype(int)
    xmax_i = np.ceil(max_coords[0]).astype(int)
    ymax_i = np.ceil(max_coords[1]).astype(int)

    # Recorta (clip) os valores ao domínio da imagem
    # O domínio vai de 0 até W-1 (para x) e 0 até H-1 (para y).
    xmin = np.clip(xmin_i, 0, W - 1)
    xmax = np.clip(xmax_i, 0, W - 1)
    ymin = np.clip(ymin_i, 0, H - 1)
    ymax = np.clip(ymax_i, 0, H - 1)

    return xmin, xmax, ymin, ymax

def _amostra_bilinear(img_float, x, y):
    """
    Amostragem bilinear em (x,y) com clamp nas bordas.
    img_float: (H,W,3) float32 [0,1] — retorna vetor (3,).
    """
    # Pega as dimensões da imagem
    H, W = img_float.shape[:2]

    # Encontra os 4 vizinhos com coordenadas inteiras
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = x1 + 1
    y2 = y1 + 1

    # Calcula os pesos da interpolação (partes fracionárias)
    # (dx, dy) estarão no intervalo [0, 1)
    dx = x - x1
    dy = y - y1

    # Faz o "clamp" (recorte) dos índices para ficarem dentro da imagem
    # Isso garante que não vamos tentar ler pixels fora das bordas.
    x1_c = np.clip(x1, 0, W - 1)
    y1_c = np.clip(y1, 0, H - 1)
    x2_c = np.clip(x2, 0, W - 1)
    y2_c = np.clip(y2, 0, H - 1)

    # Pega as cores (vetores RGB) dos 4 pixels vizinhos
    Q11 = img_float[y1_c, x1_c]  # Canto superior esquerdo (y1, x1)
    Q21 = img_float[y1_c, x2_c]  # Canto superior direito (y1, x2)
    Q12 = img_float[y2_c, x1_c]  # Canto inferior esquerdo (y2, x1)
    Q22 = img_float[y2_c, x2_c]  # Canto inferior direito (y2, x2)

    # Interpola primeiro na direção X (horizontal)
    # Interpola na linha de cima (y1)
    R1 = (1.0 - dx) * Q11 + dx * Q21
    # Interpola na linha de baixo (y2)
    R2 = (1.0 - dx) * Q12 + dx * Q22

    # Interpola na direção Y (vertical) entre os resultados R1 e R2
    P = (1.0 - dy) * R1 + dy * R2

    return P

def gera_frame(A, B, pA, pB, triangles, alfa, beta):
    """
    Gera um frame intermediário por morphing com warping por triângulos.
    - A,B: imagens (H,W,3) float32 em [0,1]
    - pA,pB: (N,2) pontos correspondentes
    - triangles: (M,3) índices de triângulos
    - alfa: controla geometria (0=A, 1=B)
    - beta:  controla mistura de cores (0=A, 1=B)
    Retorna (H,W,3) float32 em [0,1].
    """
    # Pega as dimensões da imagem
    H, W = A.shape[:2]
    
    # Calcula a geometria intermediária (p_dst)
    # Interpola os pontos de A e B usando 'alfa'
    p_dst = (1.0 - alfa) * pA + alfa * pB

    # Cria a imagem de saída (destino), inicialmente preta
    frame_out = np.zeros_like(A)

    # Itera sobre cada triângulo na malha
    for tri_indices in triangles:
        # Pega os vértices (coordenadas) do triângulo em A, B e no destino (dst)
        tri_A   = pA[tri_indices]    # (3, 2)
        tri_B   = pB[tri_indices]    # (3, 2)
        tri_dst = p_dst[tri_indices] # (3, 2)

        # Calcula o bounding box (caixa delimitadora) do triângulo de destino
        xmin, xmax, ymin, ymax = _tri_bbox(tri_dst, W, H)

        # Itera sobre todos os pixels DENTRO do bounding box
        for y_dst in range(ymin, ymax + 1):
            for x_dst in range(xmin, xmax + 1):
                pt_dst = np.array([x_dst, y_dst], dtype=np.float32)

                #Calcula as coordenadas baricêntricas de (x_dst, y_dst)
                # em relação ao triângulo de destino (tri_dst)
                w1, w2, w3 = _transf_baricentrica(pt_dst, tri_dst)

                #Verifica se o pixel está realmente dentro do triângulo
                if _check_bari(w1, w2, w3):
                    
                    #Ponto fonte em A (x_A, y_A)
                    x_A = w1 * tri_A[0][0] + w2 * tri_A[1][0] + w3 * tri_A[2][0]
                    y_A = w1 * tri_A[0][1] + w2 * tri_A[1][1] + w3 * tri_A[2][1]
                    
                    # Ponto fonte em B (x_B, y_B)
                    x_B = w1 * tri_B[0][0] + w2 * tri_B[1][0] + w3 * tri_B[2][0]
                    y_B = w1 * tri_B[0][1] + w2 * tri_B[1][1] + w3 * tri_B[2][1]

                    # Pega as cores dos pontos fonte usando
                    # interpolação bilinear (pois (x_A,y_A) e (x_B,y_B) são floats)
                    color_A = _amostra_bilinear(A, x_A, y_A)
                    color_B = _amostra_bilinear(B, x_B, y_B)

                    # Mistura as cores de A e B usando 'beta'
                    final_color = (1.0 - beta) * color_A + beta * color_B

                    # Pinta o pixel de destino
                    frame_out[y_dst, x_dst] = final_color

    return frame_out
