import logging
import voluptuous as vol
from typing import Any, Optional, Tuple
from .bjled import BJLEDInstance
from .const import DOMAIN

from homeassistant.const import CONF_MAC
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.components.light import (
    PLATFORM_SCHEMA,
    ATTR_BRIGHTNESS,
    ATTR_RGB_COLOR,
    ATTR_EFFECT,
    ColorMode,
    LightEntity,
    LightEntityFeature,
)

from homeassistant.helpers import device_registry

LOGGER = logging.getLogger(__name__)
PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({vol.Required(CONF_MAC): cv.string})

async def async_setup_entry(hass, config_entry, async_add_devices):
    instance = hass.data[DOMAIN][config_entry.entry_id]
    await instance.update()
    async_add_devices(
        [BJLEDLight(instance, config_entry.data["name"], config_entry.entry_id)]
    )

class BJLEDLight(RestoreEntity, LightEntity):
    def __init__(
        self, bjledinstance: BJLEDInstance, name: str, entry_id: str
    ) -> None:
        self._instance = bjledinstance
        self._entry_id = entry_id
        self._attr_supported_color_modes = {ColorMode.RGB}
        self._attr_supported_features = LightEntityFeature.EFFECT | LightEntityFeature.FLASH
        self._attr_brightness_step_pct = 10
        self._attr_name = name
        self._attr_unique_id = self._instance.mac
        
    @property
    def available(self):
        # return self._instance.is_on != None
        return True # We don't know if this is working or not because there is zero feedback from the light, so assume yes

    @property
    def brightness(self):
        return self._instance.brightness
    
    @property
    def rgb_color(self):
        return self._instance.rgb_color
    
    @property
    def is_on(self) -> Optional[bool]:
        return self._instance.is_on

    @property
    def effect_list(self):
        return self._instance.effect_list

    @property
    def effect(self):
        return self._instance._effect

    @property
    def supported_features(self) -> int:
        """Flag supported features."""
        return self._attr_supported_features

    @property
    def supported_color_modes(self) -> int:
        """Flag supported color modes."""
        return self._attr_supported_color_modes

    @property
    def color_mode(self):
        """Return the color mode of the light."""
        return self._instance._color_mode

    @property
    def device_info(self):
        """Return device info."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, self._instance.mac)
            },
            name=self.name,
            connections={(device_registry.CONNECTION_NETWORK_MAC, self._instance.mac)},
        )

    @property
    def should_poll(self):
        return False

    async def async_added_to_hass(self) -> None:
        """Restore last known optimistic state after HA restart."""
        await super().async_added_to_hass()
        last_state = await self.async_get_last_state()
        if last_state is None:
            return

        is_on: Optional[bool] = None
        if last_state.state in ("on", "off"):
            is_on = last_state.state == "on"

        brightness = last_state.attributes.get(ATTR_BRIGHTNESS)
        rgb_color = last_state.attributes.get(ATTR_RGB_COLOR)
        effect = last_state.attributes.get(ATTR_EFFECT)

        rgb_tuple: Optional[Tuple[int, int, int]] = None
        if isinstance(rgb_color, (list, tuple)) and len(rgb_color) == 3:
            rgb_tuple = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

        self._instance.restore_state(
            is_on=is_on,
            brightness=brightness,
            rgb_color=rgb_tuple,
            effect=effect,
        )
        self.async_write_ha_state()

    async def async_turn_on(self, **kwargs: Any) -> None:
        if not self.is_on:
            await self._instance.turn_on()
                
        if ATTR_BRIGHTNESS in kwargs and kwargs[ATTR_BRIGHTNESS] != self.brightness:
            self._brightness = kwargs[ATTR_BRIGHTNESS]
            await self._instance.set_brightness_local(kwargs[ATTR_BRIGHTNESS])
               
        if ATTR_RGB_COLOR in kwargs:
            if kwargs[ATTR_RGB_COLOR] != self.rgb_color:
                self._effect = None
                bri = kwargs[ATTR_BRIGHTNESS] if ATTR_BRIGHTNESS in kwargs else None
                await self._instance.set_rgb_color(kwargs[ATTR_RGB_COLOR], bri)
        
        if ATTR_EFFECT in kwargs:
            if kwargs[ATTR_EFFECT] != self.effect:
                self._effect = kwargs[ATTR_EFFECT]
                await self._instance.set_effect(kwargs[ATTR_EFFECT])
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._instance.turn_off()
        self.async_write_ha_state()

    async def async_set_effect(self, effect: str) -> None:
        self._effect = effect
        await self._instance.set_effect(effect)
        self.async_write_ha_state()
    
    async def async_update(self) -> None:
        await self._instance.update()
        self.async_write_ha_state()
