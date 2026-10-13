# DM1 Hackathon -- Team 9 -- Evaluation Shiny App (alternative to the Quarto dashboard)
#
# WHAT THIS FILE IS
#   A Shiny equivalent of templates/dashboard/index.qmd, in case the team prefers
#   live server-side reactivity (e.g. to later add a "predict for a hypothetical
#   patient" input form) over a static dashboard. Reads the same results/ contract --
#   see .posit/assistant/skills/dm1-results-integration/SKILL.md.
#
# RUN WITH: shiny::runApp("templates/shiny_app")

library(shiny)
library(bslib)
library(dplyr)
library(readr)
library(plotly)
library(DT)
library(rprojroot)
library(ggplot2)

root <- rprojroot::find_root(rprojroot::is_git_root)
arms <- c("Mexiletine", "Oligonucleotide", "Tideglusib")

read_arm <- function(slug, file) {
  path <- file.path(root, "results", slug, file)
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE)
}
slug_of <- function(arm) tolower(arm)

scoreboard_all  <- bind_rows(lapply(slug_of(arms), read_arm, file = "scoreboard.csv"))
covariates_all  <- bind_rows(lapply(slug_of(arms), read_arm, file = "covariates.csv"))
predictions_all <- bind_rows(lapply(slug_of(arms), read_arm, file = "predictions.csv"))

ui <- page_navbar(
  title = "DM1 Hackathon -- Team 9",
  sidebar = sidebar(
    selectInput("arm", "Treatment arm", choices = arms, selected = arms[1])
  ),
  nav_panel(
    "Model Evaluation",
    layout_columns(
      card(card_header("Predicted vs. Observed"), plotlyOutput("pred_vs_obs")),
      card(card_header("Scoreboard"), DTOutput("scoreboard")),
      col_widths = c(7, 5)
    ),
    card(card_header("Covariates: coefficients vs. importance"), DTOutput("covariates"))
  ),
  nav_panel(
    "Data Explorer",
    # Deliberately minimal: top covariates for the selected arm only.
    # Do not add a pile of filter widgets -- expand only if the team asks for more.
    card(card_header("Top covariates for the selected arm"), DTOutput("covariates_explorer"))
  )
)

server <- function(input, output, session) {

  scoreboard_arm  <- reactive(scoreboard_all  |> filter(arm == input$arm))
  covariates_arm  <- reactive(covariates_all  |> filter(arm == input$arm))
  predictions_arm <- reactive(predictions_all |> filter(arm == input$arm))

  output$pred_vs_obs <- renderPlotly({
    d <- predictions_arm()
    if (nrow(d) == 0) return(plotly_empty() |> layout(title = "No predictions yet for this arm"))
    lims <- range(c(d$observed, d$predicted), na.rm = TRUE)
    p <- ggplot(d, aes(x = observed, y = predicted, color = method)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
      geom_point(alpha = 0.7) +
      coord_equal(xlim = lims, ylim = lims) +
      labs(x = "Observed", y = "Predicted", color = "Method") +
      theme_minimal()
    ggplotly(p)
  })

  output$scoreboard <- renderDT({
    scoreboard_arm() |>
      select(Method = method, Type = method_type, Tuned = tuned, RMSE = rmse, MAE = mae, `R2` = r2) |>
      datatable(options = list(dom = "t", paging = FALSE), rownames = FALSE) |>
      formatRound(columns = c("RMSE", "MAE", "R2"), digits = 3)
  })

  output$covariates <- renderDT({
    covariates_arm() |>
      select(Method = method, Covariate = covariate, Value = value, Type = value_type, Rank = rank) |>
      arrange(Method, Rank) |>
      datatable(options = list(pageLength = 10), rownames = FALSE) |>
      formatRound(columns = "Value", digits = 3)
  })

  output$covariates_explorer <- renderDT({
    covariates_arm() |>
      filter(rank <= 5) |>
      select(Method = method, Covariate = covariate, Value = value, Rank = rank) |>
      arrange(Method, Rank) |>
      datatable(options = list(dom = "t", paging = FALSE), rownames = FALSE) |>
      formatRound(columns = "Value", digits = 3)
  })
}

shinyApp(ui, server)
