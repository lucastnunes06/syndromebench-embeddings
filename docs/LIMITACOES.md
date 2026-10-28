# Limitações e próximos experimentos

- **Sem aplicação clínica:** classificar vetores em categorias de um estudo não equivale a diagnosticar pessoas.
- **Dependência da origem dos embeddings:** não há documentação suficiente de treinamento, distribuição demográfica ou consentimento do modelo anterior para inferir generalização.
- **Validação por grupo:** separar por pessoa reduz um tipo importante de vazamento, mas não remove correlações de aquisição, lote ou laboratório.
- **Seleção de k:** o grid de 1 a 15 é finito; atingir a borda não demonstra ótimo global.
- **Pré-processamentos distintos:** primeiro experimento a realizar é manter exatamente o mesmo pré-processamento para isolar a distância.
- **Sem redistribuição automática:** qualquer divulgação de dados ou resultados de terceiros exige revisão de direitos e condições de uso.

Próximos passos verificáveis: experimentos com dados autorizados, curva de aprendizagem, comparação sob escalonamento idêntico, intervalos de confiança por pessoa e avaliação externa.
