#!/usr/bin/env julia

using Dates, DataFrames, JSON3, StructTypes
using DBInterface: close, connect, execute
using DuckDB: DB, register_data_frame
using NowcastAutoGP:
    create_nowcast_data,
    create_transformed_data,
    forecast,
    forecast_with_nowcasts,
    get_transformations,
    make_and_fit_model

"""
Input data structure for EpiAutoGP pipeline,
representing the JSON input format expected by the pipeline.
"""
struct EpiAutoGPInput
    dates::Vector{Date}
    reports::Vector{Float64}
    pathogen::String
    location::String
    target::String
    frequency::String
    ed_visit_type::String
    forecast_through::Date
    nowcast_dates::Vector{Date}
    nowcast_reports::Vector{Vector{Float64}}
end

StructType(::Type{EpiAutoGPInput}) = Struct()

const DEFAULT_ARGS = Dict{String, Any}(
    "n-particles" => 24,
    "n-mcmc" => 100,
    "n-hmc" => 50,
    "n-forecast-draws" => 2000,
    "transformation" => "boxcox",
    "smc-data-proportion" => 0.1,
)
const REQUIRED_ARGS = Set(["json-input", "output-dir"])
const INT_ARGS = Set(
    [
        "n-particles",
        "n-mcmc",
        "n-hmc",
        "n-forecast-draws",
    ]
)
const FLOAT_ARGS = Set(["smc-data-proportion"])
const STRING_ARGS = Set(["json-input", "output-dir", "transformation"])
const VALID_ARGS = union(keys(DEFAULT_ARGS), REQUIRED_ARGS)

function _parse_argument_value(name::String, value::String)
    if name in INT_ARGS
        parsed = tryparse(Int, value)
        isnothing(parsed) && throw(ArgumentError("--$name must be an integer, got '$value'"))
        return parsed
    elseif name in FLOAT_ARGS
        parsed = tryparse(Float64, value)
        isnothing(parsed) && throw(ArgumentError("--$name must be a float, got '$value'"))
        return parsed
    elseif name in STRING_ARGS
        return value
    end

    throw(ArgumentError("Unknown argument --$name"))
end

function parse_arguments(args = ARGS)
    parsed_args = copy(DEFAULT_ARGS)
    i = firstindex(args)

    while i <= lastindex(args)
        token = args[i]
        startswith(token, "--") ||
            throw(ArgumentError("Expected argument beginning with '--', got '$token'"))

        raw_name_value = token[3:end]
        equals_index = findfirst(isequal('='), raw_name_value)
        if isnothing(equals_index)
            name = replace(raw_name_value, "_" => "-")
            i == lastindex(args) &&
                throw(ArgumentError("Missing value for argument --$name"))
            value = args[i + 1]
            i += 2
        else
            name = replace(raw_name_value[begin:(equals_index - 1)], "_" => "-")
            value = raw_name_value[(equals_index + 1):end]
            i += 1
        end

        name in VALID_ARGS || throw(ArgumentError("Unknown argument --$name"))
        parsed_args[name] = _parse_argument_value(name, value)
    end

    for name in REQUIRED_ARGS
        haskey(parsed_args, name) ||
            throw(ArgumentError("Missing required argument --$name"))
    end

    return parsed_args
end

"""
JSON reader that parses the input JSON file into an EpiAutoGPInput struct.
"""
function read_data(path_to_json::String)
    return JSON3.read(read(path_to_json, String), EpiAutoGPInput)
end

################ Validation and preparation of input data for modeling
function _require(condition::Bool, message::String)
    condition || throw(ArgumentError(message))
    return nothing
end

function _validate_frequency(data::EpiAutoGPInput)
    valid_frequencies = ["daily", "epiweekly"]
    _require(
        data.frequency in valid_frequencies,
        "Frequency must be one of $(valid_frequencies), got '$(data.frequency)'",
    )
    return nothing
end

function _validate_nonnegative_finite_values(values; label::String)
    for (i, value) in enumerate(values)
        _require(
            isfinite(value) && value >= 0,
            "$label[$i] must be a non-negative finite number, got $value",
        )
    end
    return nothing
end

function _validate_observations(data::EpiAutoGPInput)
    _require(
        !isempty(data.dates) && !isempty(data.reports),
        "Empty data: dates and reports cannot be empty",
    )
    _validate_nonnegative_finite_values(data.reports; label = "reports")
    return nothing
end

function _validate_nowcasts(data::EpiAutoGPInput)
    _require(
        !isempty(data.nowcast_dates) || isempty(data.nowcast_reports),
        "Nowcast reports cannot be provided without nowcast dates",
    )

    isempty(data.nowcast_dates) && return nothing

    _require(
        issorted(data.nowcast_dates),
        "Nowcast dates must be sorted chronologically",
    )

    for (i, report_vec) in enumerate(data.nowcast_reports)
        _require(
            length(report_vec) == length(data.nowcast_dates),
            "nowcast_reports[$i] must have length $(length(data.nowcast_dates)), got $(length(report_vec))",
        )
        _validate_nonnegative_finite_values(report_vec; label = "nowcast_reports[$i]")
    end

    return nothing
end

"""
Validates the input data for the EpiAutoGP pipeline.
"""
function validate_input(data::EpiAutoGPInput)
    _validate_frequency(data)
    _validate_observations(data)
    _validate_nowcasts(data)
    return data
end

function read_and_validate_data(path_to_json::String)
    return validate_input(read_data(path_to_json))
end

################## Preparation of data for modelling and forecasting ###############

"""
Prepares the input data for modelling by:
    - Excluding nowcast dates from the stable data used to fit the base model
    - Creating nowcast data structures if nowcast reports are provided
    - Generating prediction dates across the training and forecast periods
    - Determine the data forwards and inverse transformations
"""
function prepare_for_modelling(
        input::EpiAutoGPInput,
        transformation_name::String,
        n_forecasts::Int,
    )
    stable_data_idxs = findall(date -> !(date in input.nowcast_dates), input.dates)
    stable_data_dates = input.dates[stable_data_idxs]
    stable_data_values = input.reports[stable_data_idxs]

    transformation, inv_transformation =
        get_transformations(transformation_name, Float64.(input.reports))

    nowcast_data = isempty(input.nowcast_dates) ?
        nothing :
        create_nowcast_data(input.nowcast_reports, input.nowcast_dates; transformation)

    time_step = input.frequency == "epiweekly" ? Week(1) : Day(1)
    last_training_date = input.dates[end]
    horizon_days = Dates.value(input.forecast_through - last_training_date)
    step_days = input.frequency == "epiweekly" ? 7 : 1
    _require(
        horizon_days > 0,
        "forecast_through must be after the final training date; got " *
            "$last_training_date and $(input.forecast_through)",
    )
    n_ahead, extra_days = divrem(horizon_days, step_days)
    _require(
        iszero(extra_days),
        "The distance from the final $(input.frequency) training date to " *
            "forecast_through must be a whole number of model steps; got " *
            "$last_training_date and $(input.forecast_through)",
    )
    forecast_dates = [last_training_date + i * time_step for i in 1:n_ahead]
    prediction_dates = vcat(input.dates, forecast_dates)

    n_forecasts_per_nowcast = isnothing(nowcast_data) ?
        n_forecasts :
        max(1, n_forecasts ÷ length(nowcast_data))

    return (;
        stable_data_dates,
        stable_data_values,
        nowcast_data,
        prediction_dates,
        n_forecasts_per_nowcast,
        transformation,
        inv_transformation,
    )
end

"""
Fit the ensemble GP model using the stable data (not expected to be revised
by nowcasts) and return the fitted model object.
"""
function fit_base_model(
        dates::Vector{Date},
        values::Vector{Float64};
        transformation,
        n_particles::Int = 24,
        smc_data_proportion::Float64 = 0.1,
        n_mcmc::Int = 50,
        n_hmc::Int = 50,
    )
    transformed_data = create_transformed_data(dates, values; transformation)
    return make_and_fit_model(
        transformed_data;
        n_particles = n_particles,
        smc_data_proportion = smc_data_proportion,
        n_mcmc = n_mcmc,
        n_hmc = n_hmc,
    )
end

"""
Generate forecasts from the fitted model, using nowcast data if provided.
If nowcast data is provided, generate forecasts for each nowcast scenario and pool the results.
"""
function _do_forecasts(
        nowcast_data,
        base_model,
        prediction_dates,
        n_forecasts_per_nowcast::Int;
        inv_transformation,
    )
    return forecast_with_nowcasts(
        base_model,
        nowcast_data,
        prediction_dates,
        n_forecasts_per_nowcast;
        inv_transformation = inv_transformation,
    )
end

function _do_forecasts(
        nowcast_data::Nothing,
        base_model,
        prediction_dates,
        n_forecasts_per_nowcast::Int;
        inv_transformation,
    )
    return forecast(
        base_model,
        prediction_dates,
        n_forecasts_per_nowcast;
        inv_transformation = inv_transformation,
    )
end

function forecast_with_nowcastautogp(input::EpiAutoGPInput, args::Dict{String, Any})
    model_info = prepare_for_modelling(
        input,
        args["transformation"],
        args["n-forecast-draws"],
    )
    base_model = fit_base_model(
        model_info.stable_data_dates,
        model_info.stable_data_values;
        transformation = model_info.transformation,
        n_particles = args["n-particles"],
        smc_data_proportion = args["smc-data-proportion"],
        n_mcmc = args["n-mcmc"],
        n_hmc = args["n-hmc"],
    )
    forecasts = _do_forecasts(
        model_info.nowcast_data,
        base_model,
        model_info.prediction_dates,
        model_info.n_forecasts_per_nowcast;
        inv_transformation = model_info.inv_transformation,
    )

    return (; prediction_dates = model_info.prediction_dates, forecasts = forecasts)
end

################## Formatting and saving forecast output ###############
function create_forecast_df(results::NamedTuple)
    return mapreduce(vcat, enumerate(eachcol(results.forecasts))) do (draw, sampled_values)
        DataFrame(
            :date => results.prediction_dates,
            Symbol(".value") => sampled_values,
            Symbol(".draw") => fill(Int32(draw), length(sampled_values)),
        )
    end
end

function _quote_duckdb_string(value::AbstractString)
    return "'" * replace(value, "'" => "''") * "'"
end

function _write_parquet_with_duckdb(path::AbstractString, table)
    con = connect(DB, ":memory:")
    return try
        register_data_frame(con, table, "forecast_samples")
        execute(
            con,
            "COPY forecast_samples TO $(_quote_duckdb_string(path)) (FORMAT parquet)",
        )
    finally
        close(con)
    end
end

function create_forecast_output(
        input::EpiAutoGPInput,
        results::NamedTuple,
        output_dir::String;
        save_output::Bool,
    )
    forecast_df = create_forecast_df(results)

    variable_name = if input.target == "nhsn"
        "observed_hospital_admissions"
    else
        Dict(
            "observed" => "observed_ed_visits",
            "other" => "other_ed_visits",
            "pct" => "prop_disease_ed_visits",
        )[input.ed_visit_type]
    end

    if input.ed_visit_type == "pct" && input.target == "nssp"
        forecast_df[!, Symbol(".value")] = forecast_df[!, Symbol(".value")] ./ 100.0
    end

    forecast_df[!, Symbol(".variable")] .= variable_name
    forecast_df[!, :resolution] .= input.frequency
    forecast_df[!, :geo_value] .= input.location
    forecast_df[!, :disease] .= input.pathogen

    if save_output
        parquet_path = joinpath(output_dir, "samples.parquet")
        mkpath(dirname(parquet_path))
        _write_parquet_with_duckdb(parquet_path, forecast_df)
        println("Saved pipeline forecast samples to $parquet_path")
    end

    return forecast_df
end

function main()
    return try
        args = parse_arguments()
        input_data = read_and_validate_data(args["json-input"])
        results = forecast_with_nowcastautogp(input_data, args)
        create_forecast_output(
            input_data,
            results,
            args["output-dir"];
            save_output = true,
        )
    catch e
        println(stderr, "NowcastAutoGP pipeline run failed:")
        showerror(stderr, e, catch_backtrace())
        println(stderr)
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
