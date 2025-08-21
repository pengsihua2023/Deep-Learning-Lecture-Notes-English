
```latex
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{xcolor}

% 页面设置
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}

% Python 代码高亮设置
\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{green!50!black},
    stringstyle=\color{red},
    showstringspaces=false,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny,
}

\title{Long Short-Term Memory (LSTM) Study Notes}
\author{}
\date{}

\begin{document}

\maketitle

\section{Introduction}

\textbf{Long Short-Term Memory (LSTM)} is a special type of Recurrent Neural Network (RNN) designed to solve the \textbf{vanishing gradient problem} of standard RNNs.

\begin{itemize}
    \item \textbf{Strengths:} Capable of learning and remembering long-range dependencies in sequential data.
    \item \textbf{Applications:} Widely used in \textbf{speech recognition}, \textbf{time series forecasting}, \textbf{text generation}, \textbf{machine translation}, and more.
\end{itemize}

\section{Core Concept}

LSTM introduces a \textbf{gating mechanism} that regulates the flow of information:

\begin{itemize}
    \item \textbf{Forget Gate:} Decides how much past information should be discarded.
    \item \textbf{Input Gate:} Controls how much new information is added to the cell state.
    \item \textbf{Output Gate:} Determines the output of the current hidden state.
\end{itemize}

Through these gates, LSTMs effectively retain useful information and mitigate gradient vanishing.

\section{LSTM Cell Structure}

At each time step $t$, an LSTM cell receives:

\begin{itemize}
    \item the previous hidden state $h_{t-1}$,
    \item the previous cell state $c_{t-1}$,
    \item and the current input $x_t$.
\end{itemize}

The structure consists of:

\begin{itemize}
    \item Input Gate
    \item Forget Gate
    \item Output Gate
    \item Cell State
\end{itemize}

\section{Mathematical Formulation}

\subsection{Forget Gate}

\[
f_t = \sigma\big( W_f [h_{t-1}, x_t] + b_f \big)
\]

\textit{Controls what fraction of the previous cell state is retained.}

\subsection{Input Gate}

\[
i_t = \sigma\big( W_i [h_{t-1}, x_t] + b_i \big)
\]

\[
\tilde{c}_t = \tanh\big( W_c [h_{t-1}, x_t] + b_c \big)
\]

\textit{Determines what new information is stored in the cell state.}

\subsection{Cell State Update}

\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\]

\textit{Updates the cell state by combining retained and new information.}

\subsection{Output Gate}

\[
o_t = \sigma\big( W_o [h_{t-1}, x_t] + b_o \big)
\]

\[
h_t = o_t \odot \tanh(c_t)
\]

\textit{Controls the output of the hidden state.}

\section{Parameters}

\begin{itemize}
    \item $W_f, W_i, W_c, W_o$: Weight matrices
    \item $b_f, b_i, b_c, b_o$: Bias terms
    \item $h_t$: Hidden state
    \item $c_t$: Cell state
\end{itemize}

\section{PyTorch Implementation}

\begin{lstlisting}
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_new = nn.Linear(hidden_size, 20)
        self.fc = nn.Linear(20, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc_new(out)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
\end{lstlisting}

\section{Training \& Results}

\begin{itemize}
    \item \textbf{Loss function:} Binary Cross-Entropy (BCELoss)
    \item \textbf{Optimizer:} Adam
    \item \textbf{Dataset:} UCI HAR (Human Activity Recognition), binary classification (Walking vs. Non-Walking)
\end{itemize}

\textbf{Training Results:}

\begin{itemize}
    \item Loss converges rapidly
    \item Final test accuracy: \textbf{98.71\%}
\end{itemize}

\section{Key Takeaways}

\begin{itemize}
    \item LSTM is a powerful extension of RNN for handling \textbf{long-term dependencies}.
    \item The \textbf{gating mechanism} (forget, input, output) is crucial for controlling information flow.
    \item Practical implementations (e.g., PyTorch) are straightforward and widely used in real-world sequence modeling tasks.
\end{itemize}

\end{document}
```

---



