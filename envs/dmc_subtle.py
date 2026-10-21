from dm_control.rl import control
from dm_control.suite import ball_in_cup, cartpole, finger, point_mass, reacher
from lxml import etree

SCALES = {
    "ball_in_cup_catch_subtle": 1 / 12,
    "point_mass_subtle": 1 / 6,
    "finger_turn_subtle": 1 / 2,
    "reacher_subtle": 1 / 3,
    "cartpole_subtle": 1 / 20,
}


def _modify_xml_element_size(xml_string, element_name, new_size_str, element_type="geom"):
    """Parses an XML string, modifies the size of a specified element, and returns the modified XML string."""
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    element = mjcf.find(f".//{element_type}[@name='{element_name}']")
    if element is not None:
        element.set("size", new_size_str)
    else:
        raise ValueError(f"Element '{element_name}' of type '{element_type}' not found in XML.")

    return etree.tostring(mjcf, pretty_print=True)


# --- Reacher ---
def reacher_subtle(time_limit=reacher._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns reacher with a custom target size."""
    _DEFAULT_SIZE = reacher._SMALL_TARGET
    physics = reacher.Physics.from_xml_string(*reacher.get_model_and_assets())
    task = reacher.Reacher(target_size=_DEFAULT_SIZE * SCALES["reacher_subtle"], random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


# --- Finger ---
def finger_turn_subtle(time_limit=finger._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the easy Turn task with a custom target size."""
    _DEFAULT_SIZE = finger._HARD_TARGET_SIZE
    physics = finger.Physics.from_xml_string(*finger.get_model_and_assets())
    task = finger.Turn(target_radius=_DEFAULT_SIZE * SCALES["finger_turn_subtle"], random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=finger._CONTROL_TIMESTEP, **environment_kwargs
    )


# --- Point-mass ---
def point_mass_subtle(time_limit=point_mass._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the point_mass hard task with a custom target size by modifying the XML."""
    _DEFAULT_TARGET_SIZE = 0.015  # Default is .015
    _DEFAULT_AGENT_SIZE = 0.01  # Default is .01
    xml_string, assets = point_mass.get_model_and_assets()
    modified_xml = _modify_xml_element_size(
        xml_string, "target", str(_DEFAULT_TARGET_SIZE * SCALES["point_mass_subtle"])
    )
    modified_xml = _modify_xml_element_size(
        modified_xml, "pointmass", str(_DEFAULT_AGENT_SIZE * SCALES["point_mass_subtle"])
    )
    physics = point_mass.Physics.from_xml_string(modified_xml, assets)
    task = point_mass.PointMass(randomize_gains=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


# --- Ball-in-cup ---
def ball_in_cup_catch_subtle(time_limit=ball_in_cup._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Ball-in-Cup task with a custom ball size and string width by modifying the XML."""
    _DEFAULT_BALL_SIZE = 0.025  # Default is .025
    _DEFAULT_STRING_WIDTH = 0.003  # Default is 0.003

    xml_string, assets = ball_in_cup.get_model_and_assets()

    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    # Modify ball size
    ball = mjcf.find(".//geom[@name='ball']")
    if ball is not None:
        ball.set("size", str(_DEFAULT_BALL_SIZE * SCALES["ball_in_cup_catch_subtle"]))
    else:
        raise ValueError("Element 'ball' not found in XML.")

    # Modify string width
    string_tendon = mjcf.find(".//tendon/spatial[@name='string']")
    if string_tendon is not None:
        string_tendon.set("width", str(_DEFAULT_STRING_WIDTH * SCALES["ball_in_cup_catch_subtle"]))
    else:
        raise ValueError("Tendon 'string' not found in XML.")

    modified_xml = etree.tostring(mjcf, pretty_print=True)

    physics = ball_in_cup.Physics.from_xml_string(modified_xml, assets)
    task = ball_in_cup.BallInCup(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=ball_in_cup._CONTROL_TIMESTEP, **environment_kwargs
    )


# --- Cartpole ---
def _get_cartpole_subtle_physics(random=None):
    """Returns a Physics instance with a thinned pole."""
    xml_string, assets = cartpole.get_model_and_assets()
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    # Find the default pole geom to get the original size and modify it.
    default_pole_geom = mjcf.find(".//default/default[@class='pole']/geom")
    if default_pole_geom is None:
        raise ValueError("Could not find default geom for class 'pole' in the cartpole XML model.")

    radius_str = default_pole_geom.get("size")
    if radius_str is None:
        raise ValueError("Default pole geom does not have a 'size' attribute.")
    radius = float(radius_str)

    # Calculate new radius and set it on the default class.
    # This will be inherited by all geoms with childclass="pole".
    new_radius = radius * SCALES["cartpole_subtle"]
    default_pole_geom.set("size", str(new_radius))

    modified_xml = etree.tostring(mjcf, pretty_print=True)
    return cartpole.Physics.from_xml_string(modified_xml, assets)


def cartpole_swingup_subtle(time_limit=cartpole._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Cartpole Swing-Up task with a thinned pole."""
    physics = _get_cartpole_subtle_physics(random=random)
    task = cartpole.Balance(swing_up=True, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)
