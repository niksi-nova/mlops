from src.monitoring.performance import mae, mae_degradation_pct, performance_status


def test_mae_value():
    assert mae([10, 20, 30], [12, 18, 33]) == (2 + 2 + 3) / 3


def test_degradation_pct():
    assert round(mae_degradation_pct(36.0, 30.6), 1) == 17.6
    assert round(mae_degradation_pct(33.0, 30.6), 1) == 7.8


def test_performance_status():
    assert performance_status(17.6, 15.0) == "ALERT"
    assert performance_status(10.0, 15.0) == "WARNING"
    assert performance_status(3.0, 15.0) == "OK"
    assert performance_status(-5.0, 15.0) == "OK"
