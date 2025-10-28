import cv2
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Frame, StringVar, Scale, HORIZONTAL, SUNKEN, W, simpledialog
from tkinter import ttk
from PIL import Image, ImageTk
import math


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Processador de Imagens - UniSALESIANO")
        self.root.geometry("1280x960")

        self.img_original = None
        self.img_processada = None

        self.metodos = {
            "Média": self.metodo_media,
            "Luminância": self.metodo_luminancia,
            "Canal Vermelho": self.metodo_canal_vermelho,
            "Canal Verde": self.metodo_canal_verde,
            "Canal Azul": self.metodo_canal_azul
        }
        self.metodo_selecionado = StringVar(value="Luminância")
        self.limiar_binarizacao = 128

        self.criar_interface()

    def criar_interface(self):
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill='both', expand=True)

        control_frame = Frame(main_frame, width=280)
        control_frame.pack(side='left', fill='y')

        img_frame = Frame(main_frame)
        img_frame.pack(side='left', fill='both', expand=True)

        original_frame = Frame(img_frame)
        original_frame.pack(side='left', fill='both', expand=True)

        processada_frame = Frame(img_frame)
        processada_frame.pack(side='right', fill='both', expand=True)

        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ---------- Controles do projeto ----------
        ttk.Button(scrollable_frame, text="Carregar Imagem", command=self.carregar_imagem).pack(pady=5, fill='x')

        info_frame = ttk.LabelFrame(scrollable_frame, text="Informações da Imagem")
        info_frame.pack(pady=5, fill='x')
        self.lbl_tamanho = ttk.Label(info_frame, text="Tamanho: -")
        self.lbl_tamanho.pack(anchor='w')
        self.lbl_altura = ttk.Label(info_frame, text="Altura: -")
        self.lbl_altura.pack(anchor='w')
        self.lbl_largura = ttk.Label(info_frame, text="Largura: -")
        self.lbl_largura.pack(anchor='w')

        conv_frame = ttk.LabelFrame(scrollable_frame, text="Conversão Tons de Cinza")
        conv_frame.pack(pady=5, fill='x')
        for metodo in self.metodos:
            ttk.Radiobutton(conv_frame, text=metodo, variable=self.metodo_selecionado, value=metodo).pack(anchor='w')
        ttk.Button(conv_frame, text="Converter", command=self.converter_imagem).pack(pady=3, fill='x')

        bin_frame = ttk.LabelFrame(scrollable_frame, text="Binarização")
        bin_frame.pack(pady=5, fill='x')
        self.scale_limiar = Scale(bin_frame, from_=0, to=255, orient=HORIZONTAL, command=self.atualizar_limiar)
        self.scale_limiar.set(self.limiar_binarizacao)
        self.scale_limiar.pack(pady=3, fill='x')
        ttk.Button(bin_frame, text="Aplicar Binarização", command=self.binarizar_imagem).pack(fill='x')

        baixa_frame = ttk.LabelFrame(scrollable_frame, text="Filtros Passa-Baixa")
        baixa_frame.pack(pady=5, fill='x')
        ttk.Button(baixa_frame, text="Média", command=self.aplicar_filtro_media).pack(fill='x')
        ttk.Button(baixa_frame, text="Mediana", command=self.aplicar_filtro_mediana).pack(fill='x')
        ttk.Button(baixa_frame, text="Gaussiano", command=self.aplicar_filtro_gaussiano).pack(fill='x')

        alta_frame = ttk.LabelFrame(scrollable_frame, text="Filtros Passa-Alta")
        alta_frame.pack(pady=5, fill='x')
        ttk.Button(alta_frame, text="Laplaciano", command=self.aplicar_filtro_laplaciano).pack(fill='x')
        ttk.Button(alta_frame, text="Sobel", command=self.aplicar_filtro_sobel).pack(fill='x')
        ttk.Button(alta_frame, text="Prewitt", command=self.aplicar_filtro_prewitt).pack(fill='x')
        ttk.Button(alta_frame, text="Roberts", command=self.aplicar_filtro_roberts).pack(fill='x')

        realce_frame = ttk.LabelFrame(scrollable_frame, text="Realce de Intensidade")
        realce_frame.pack(pady=5, fill='x')
        ttk.Button(realce_frame, text="Negativo", command=self.negativo).pack(fill='x')
        ttk.Button(realce_frame, text="Logarítmico", command=self.logaritmico).pack(fill='x')
        ttk.Button(realce_frame, text="Potência (Gama)", command=self.potencia).pack(fill='x')
        ttk.Button(realce_frame, text="Ajuste de Contraste", command=self.ajuste_contraste).pack(fill='x')

        ttk.Button(scrollable_frame, text="Equalização de Histograma", command=self.equalizar).pack(pady=5, fill='x')

        fatiamento_frame = ttk.LabelFrame(scrollable_frame, text="Fatiamento")
        fatiamento_frame.pack(pady=5, fill='x')
        ttk.Button(fatiamento_frame, text="Por Intervalo", command=self.fatiamento_intervalo).pack(fill='x')
        ttk.Button(fatiamento_frame, text="Plano de Bits", command=self.fatiamento_bits).pack(fill='x')
        ttk.Button(fatiamento_frame, text="Visualizar Todos os Planos de Bits", command=self.visualizar_todos_planos_bits).pack(fill='x')

        ttk.Button(scrollable_frame, text="Gerar Histograma", command=self.gerar_histograma).pack(pady=5, fill='x')

        ttk.Button(scrollable_frame, text="Salvar Imagem", command=self.salvar_imagem).pack(pady=10, fill='x')

        # ---------- Imagens ----------
        self.lbl_img_original = ttk.Label(original_frame, text="Imagem Original")
        self.lbl_img_original.pack()
        self.canvas_original = Label(original_frame)
        self.canvas_original.pack()

        self.lbl_img_processada = ttk.Label(processada_frame, text="Imagem Processada")
        self.lbl_img_processada.pack()
        self.canvas_processada = Label(processada_frame)
        self.canvas_processada.pack()

        self.status_bar = ttk.Label(self.root, text="Pronto", relief=SUNKEN, anchor=W)
        self.status_bar.pack(side='bottom', fill='x')

    # ----------------------- Funções -----------------------
    def carregar_imagem(self):
        caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])
        if caminho:
            self.img_original = cv2.imread(caminho)
            if self.img_original is not None:
                self.img_processada = None
                h, w = self.img_original.shape[:2]
                self.lbl_tamanho.config(text=f"Tamanho: {h * w} pixels")
                self.lbl_altura.config(text=f"Altura: {h} pixels")
                self.lbl_largura.config(text=f"Largura: {w} pixels")
                self.mostrar_imagens()
                self.status_bar.config(text=f"Imagem carregada")

    def mostrar_imagens(self):
        if self.img_original is None:
            return
        largura_max, altura_max = 400, 400
        img_original_rgb = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
        img_original_pil = Image.fromarray(img_original_rgb)
        ratio = min(largura_max / img_original_pil.width, altura_max / img_original_pil.height)
        size = (int(img_original_pil.width * ratio), int(img_original_pil.height * ratio))
        img_original_pil = img_original_pil.resize(size, Image.LANCZOS)
        img_original_tk = ImageTk.PhotoImage(img_original_pil)
        self.canvas_original.config(image=img_original_tk)
        self.canvas_original.image = img_original_tk

        if self.img_processada is not None:
            if len(self.img_processada.shape) == 2:
                img_proc_rgb = cv2.cvtColor(self.img_processada, cv2.COLOR_GRAY2RGB)
            else:
                img_proc_rgb = cv2.cvtColor(self.img_processada, cv2.COLOR_BGR2RGB)
            img_proc_pil = Image.fromarray(img_proc_rgb)
            img_proc_pil = img_proc_pil.resize(size, Image.LANCZOS)
            img_proc_tk = ImageTk.PhotoImage(img_proc_pil)
            self.canvas_processada.config(image=img_proc_tk)
            self.canvas_processada.image = img_proc_tk
        else:
            self.canvas_processada.config(image='')
            self.canvas_processada.image = None

    def atualizar_limiar(self, valor):
        self.limiar_binarizacao = int(valor)

    def metodo_media(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    def metodo_luminancia(self, img):
        return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)

    def metodo_canal_vermelho(self, img):
        return img[:, :, 0]

    def metodo_canal_verde(self, img):
        return img[:, :, 1]

    def metodo_canal_azul(self, img):
        return img[:, :, 2]

    def converter_imagem(self):
        metodo = self.metodo_selecionado.get()
        conversor = self.metodos[metodo]
        img_rgb = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
        self.img_processada = conversor(img_rgb)
        self.mostrar_imagens()
        self.status_bar.config(text=f"Imagem convertida usando {metodo}")

    def binarizar_imagem(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        _, img_binaria = cv2.threshold(img_cinza, self.limiar_binarizacao, 255, cv2.THRESH_BINARY)
        self.img_processada = img_binaria
        self.mostrar_imagens()

    def aplicar_filtro_media(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        self.img_processada = cv2.blur(img_cinza, (3, 3))
        self.mostrar_imagens()

    def aplicar_filtro_mediana(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        self.img_processada = cv2.medianBlur(img_cinza, 3)
        self.mostrar_imagens()

    def aplicar_filtro_gaussiano(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        self.img_processada = cv2.GaussianBlur(img_cinza, (5, 5), 0)
        self.mostrar_imagens()

    def aplicar_filtro_laplaciano(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(img_cinza, cv2.CV_64F)
        self.img_processada = cv2.convertScaleAbs(lap)
        self.mostrar_imagens()

    def aplicar_filtro_sobel(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(img_cinza, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_cinza, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        self.img_processada = cv2.convertScaleAbs(sobel)
        self.mostrar_imagens()

    def aplicar_filtro_prewitt(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(img_cinza, -1, kernelx)
        prewitty = cv2.filter2D(img_cinza, -1, kernely)
        self.img_processada = cv2.convertScaleAbs(prewittx + prewitty)
        self.mostrar_imagens()

    def aplicar_filtro_roberts(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(img_cinza, -1, kernelx)
        robertsy = cv2.filter2D(img_cinza, -1, kernely)
        self.img_processada = cv2.convertScaleAbs(robertsx + robertsy)
        self.mostrar_imagens()

    def negativo(self):
        img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        self.img_processada = 255 - img
        self.mostrar_imagens()

    def logaritmico(self):
        img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        c = 255 / np.log(1 + np.max(img))
        self.img_processada = cv2.convertScaleAbs(c * np.log(1 + img))
        self.mostrar_imagens()

    def potencia(self):
        gama = simpledialog.askfloat("Potência (Gama)", "Informe o valor de gama (ex.: 0.5, 2):", minvalue=0.1)
        if gama:
            img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            img = np.power(img, gama)
            self.img_processada = cv2.convertScaleAbs(img * 255)
            self.mostrar_imagens()

    def ajuste_contraste(self):
        alfa = simpledialog.askfloat("Ajuste de Contraste", "Informe o valor de alfa (ex.: 1.5 para mais contraste):", minvalue=0.1)
        if alfa:
            img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            self.img_processada = cv2.convertScaleAbs(img, alpha=alfa, beta=0)
            self.mostrar_imagens()

    def equalizar(self):
        img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        self.img_processada = cv2.equalizeHist(img)
        self.mostrar_imagens()

    def fatiamento_intervalo(self):
        img_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        min_val = simpledialog.askinteger("Fatiamento", "Valor mínimo (0 a 255):", minvalue=0, maxvalue=255)
        max_val = simpledialog.askinteger("Fatiamento", "Valor máximo (0 a 255):", minvalue=0, maxvalue=255)
        if min_val is not None and max_val is not None and min_val < max_val:
            img_fatiado = np.zeros_like(img_cinza)
            img_fatiado[(img_cinza >= min_val) & (img_cinza <= max_val)] = 255
            self.img_processada = img_fatiado
            self.mostrar_imagens()
            self.status_bar.config(text=f"Fatiamento aplicado: intervalo {min_val} a {max_val}")
        else:
            self.status_bar.config(text="Intervalo inválido para fatiamento")

    def fatiamento_bits(self):
        img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        min_plano = simpledialog.askinteger("Plano de Bits", "Plano mínimo (0-7):", minvalue=0, maxvalue=7)
        max_plano = simpledialog.askinteger("Plano de Bits", "Plano máximo (0-7):", minvalue=0, maxvalue=7)

        if min_plano is not None and max_plano is not None:
            if min_plano > max_plano:
                min_plano, max_plano = max_plano, min_plano

        resultado = np.zeros_like(img, dtype=np.uint8)
        for i in range(min_plano, max_plano + 1):
            resultado |= ((img & (1 << i)) >> i) << i
        self.img_processada = resultado
        self.mostrar_imagens()

    def visualizar_todos_planos_bits(self):
        img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        planos = []

        for i in range(8):
            plano = (img & (1 << i)) >> i
            plano *= 255  # Escala para visualização
            planos.append(plano.astype(np.uint8))

        linha1 = np.hstack(planos[:4])
        linha2 = np.hstack(planos[4:])
        grid = np.vstack([linha1, linha2])

        cv2.imshow("Planos de Bits (0-7)", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gerar_histograma(self):
        if len(self.img_original.shape) == 3:
            img_original_cinza = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        else:
            img_original_cinza = self.img_original

        if self.img_processada is not None:
            if len(self.img_processada.shape) == 3:
                img_processada_cinza = cv2.cvtColor(self.img_processada, cv2.COLOR_BGR2GRAY)
            else:
                img_processada_cinza = self.img_processada
        else:
            img_processada_cinza = None

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Histograma - Imagem Original")
        plt.xlabel("Intensidade de Pixel (0-255)")
        plt.ylabel("Quantidade de Pixels")
        plt.hist(img_original_cinza.ravel(), bins=256, range=(0, 256), color='gray', edgecolor='black')
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.subplot(1, 2, 2)
        plt.title("Histograma - Imagem Processada")
        plt.xlabel("Intensidade de Pixel (0-255)")
        plt.ylabel("Quantidade de Pixels")
        if img_processada_cinza is not None:
            plt.hist(img_processada_cinza.ravel(), bins=256, range=(0, 256), color='blue', edgecolor='black')
        else:
            plt.text(0.5, 0.5, 'Imagem não processada', fontsize=12, ha='center', va='center')
            plt.xlim(0, 256)
            plt.ylim(0, 100)

        plt.tight_layout()
        plt.show()

        self.status_bar.config(text="Histogramas gerados.")



    def salvar_imagem(self):
        if self.img_processada is not None:
            caminho = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
            if caminho:
                cv2.imwrite(caminho, self.img_processada)


# ----------- Executar -----------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()