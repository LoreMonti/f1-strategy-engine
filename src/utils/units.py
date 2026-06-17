# =========================================================
# Unit conversion utilities
# =========================================================


def ms_to_kmh(speed_ms: float) -> float:
    """Convert speed from m/s to km/h."""
    return speed_ms * 3.6


def kmh_to_ms(speed_kmh: float) -> float:
    """Convert speed from km/h to m/s."""
    return speed_kmh / 3.6


def rpm_to_rad_s(rpm: float) -> float:
    """Convert rotational speed from RPM to rad/s."""
    import math
    return rpm * 2.0 * math.pi / 60.0


def rad_s_to_rpm(rad_s: float) -> float:
    """Convert rotational speed from rad/s to RPM."""
    import math
    return rad_s * 60.0 / (2.0 * math.pi)


def deg_to_rad(degrees: float) -> float:
    """Convert angle from degrees to radians."""
    import math
    return math.radians(degrees)


def rad_to_deg(radians: float) -> float:
    """Convert angle from radians to degrees."""
    import math
    return math.degrees(radians)


def w_to_kw(watts: float) -> float:
    """Convert power from W to kW."""
    return watts / 1000.0


def kw_to_w(kilowatts: float) -> float:
    """Convert power from kW to W."""
    return kilowatts * 1000.0


def nm_to_kgm(torque_nm: float) -> float:
    """Convert torque from Nm to kgf·m."""
    from src.utils.constants import GRAVITY
    return torque_nm / GRAVITY


def m_to_km(distance_m: float) -> float:
    """Convert distance from m to km."""
    return distance_m / 1000.0


def km_to_m(distance_km: float) -> float:
    """Convert distance from km to m."""
    return distance_km * 1000.0
