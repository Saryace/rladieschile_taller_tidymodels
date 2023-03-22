
# Instalación de librerias ------------------------------------------------

# install.packages(c("tidymodels",
#                    "remotes",
#                    "workflows",
#                    "workflowsets"))
# 
# remotes::install_github("cienciadedatos/datos")


# Cargamos las librerías --------------------------------------------------

library(tidyverse) # base para analisis de datos
library(tidymodels) # base para modelacion
library(workflows) # funciones para concatenar el flujo de trabajo
library(workflowsets) # workflows: funciones extra

# Cargamos los datos ------------------------------------------------------

pinguinos <- datos::pinguinos

pinguinos

# Hacemos un análisis superficial -----------------------------------------

glimpse(pinguinos)

# Plot variables ----------------------------------------------------------

pinguinos %>%
  drop_na() %>% # removemos filas con datos ausentes
  ggplot(aes(x = masa_corporal_g,
             y = largo_aleta_mm)) +
  geom_point(alpha = 0.3) + 
  labs(
    x = "masa corporal en gramos",
    y = "largo aleta",
    title = "Análisis exploratorio masa pinguinos",
    subtitle = "Masa vs largo aleta"
  ) +
  theme_bw() # estilo del plot

# Partición de datos ------------------------------------------------------

# seed es para asegurarnos de que obtenemos los mismos resultados para la aleatorización.

set.seed(2023)

# Dividimos los datos en 80% entrenamiento y 20% testeo

pinguinos_division <- initial_split(pinguinos,
                                    prop = 0.80,
                                    strata = masa_corporal_g)

# Definimos los sets

pinguinos_entrenamiento <- training(pinguinos_division)

pinguinos_testeo  <-  testing(pinguinos_division)

# Creamos una receta ------------------------------------------------------

receta_pinguinos <-
  recipe(masa_corporal_g ~ largo_aleta_mm + largo_pico_mm,
         data = pinguinos_entrenamiento) %>%
  step_impute_mean(masa_corporal_g,
                   largo_aleta_mm,
                   largo_pico_mm)

# Seteamos el modelo ------------------------------------------------------

modelo_lm <- # modelo lineal sin hiperparametros
  parsnip::linear_reg() %>%
  parsnip::set_engine("lm")


# Creamos un workflow para usar el modelo y receta ------------------------

workflow_lm <-
  workflow() %>% 
  workflows::add_recipe(receta_pinguinos) %>% 
  workflows::add_model(modelo_lm) 

ajuste_lm <- fit(workflow_lm, pinguinos_entrenamiento)

ajuste_lm

# evaluamos el modelo

ajuste_lm %>% tidy()


# Creamos dos modelo glm: uno con paremetros fijos y otro tuneable --------


# Parametros fijos: penalty y mixture -------------------------------------

modelo_glm <- 
  parsnip::linear_reg(penalty = 0.1,
                      mixture = 0.95) %>%
  parsnip::set_engine("glmnet")

modelo_glm

# Parametros tuneables ----------------------------------------------------

# primero hay que hacer una grilla

penalty_tune <- 10 ^ seq(-3, 0,
                         length.out = 10)

grilla <- crossing(penalty = penalty_tune,
                   mixture = c(0.1, 1.0))


# Ahora creamos el modelo tuneable ----------------------------------------

modelo_glm_tune <- 
  linear_reg(penalty = tune(),
             mixture = tune()) %>% 
  set_engine("glmnet",
             path_values = penalty_tune)

# Validacion cruzada ------------------------------------------------------

pinguinos_folds <- vfold_cv(pinguinos_entrenamiento, v = 5)

pinguinos_folds

# modelo glm() con hiperparametros tuneables

workflow_glm_tune <- 
  workflow() %>% 
  workflows::add_recipe(receta_pinguinos) %>% 
  workflows::add_model(modelo_glm_tune) 

glm_tuned <- tune_grid(
  # el workflow para tunear
  workflow_glm_tune, 
  # los "folds"
  resamples = pinguinos_folds,
  # La grilla de hiperparametros
  grid = grilla)

# Listo el tuneo, obtener el mejor set de parametros ----------------------

autoplot(glm_tuned)

glm_best <- select_best(glm_tuned, metric = "rmse")

glm_best


# Ahora usamos los mejores parametros en testeo ---------------------------

glm_best_workflow <- 
  workflow_glm_tune %>%
  finalize_workflow(tibble(penalty = 1, 
                           mixture = 0.1)) %>%
  fit(data = pinguinos_entrenamiento)

glm_best_workflow 

last_fit_glm <- last_fit(glm_best_workflow, pinguinos_division)

last_fit_glm %>% collect_metrics()


# Evaluamos graficamente el set de testeo ---------------------------------

last_fit_glm %>% 
  collect_predictions() %>%
  ggplot(aes(x = masa_corporal_g,
             y = .pred)) +
  geom_abline(lty = 2, color = "purple", size = 2) +
  geom_point(alpha = 0.3) +
  labs(
    x = "masa corporal en gramos",
    y = "masa predicha",
    title = "Predicción masa pinguinos",
    subtitle = "Análisis en el set de testeo usando modelo glm"
  ) +
  theme_bw()


