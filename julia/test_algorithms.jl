using JSON

function test_algorithms()
    json_data = JSON.parsefile("discovered.json")
    return json_data
end

function run_matrix_test()
    algo = test_algorithms()
    println("Testing Reinforcement Learning Discovered Algorithm for ", algo["matrix-sizes"])
end

run_matrix_test()