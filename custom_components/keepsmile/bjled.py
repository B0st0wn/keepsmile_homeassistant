import asyncio
import contextlib
import time
from homeassistant.components import bluetooth
from homeassistant.components.light import (ColorMode)
from bleak.backends.device import BLEDevice
from bleak.backends.service import BleakGATTCharacteristic, BleakGATTServiceCollection
from bleak.exc import BleakDBusError
from bleak_retry_connector import BLEAK_RETRY_EXCEPTIONS as BLEAK_EXCEPTIONS
from bleak_retry_connector import (
    BleakClientWithServiceCache,
    #BleakError,
    BleakNotFoundError,
    #ble_device_has_changed,
    establish_connection,
)
from cheshire.compiler.compiler import StateCompiler
from cheshire.compiler.state import LightState
from cheshire.generic.command import *
from cheshire.hal.devices import device_profile_from_ble_device
from cheshire.communication.transmitter import Transmitter
from typing import Any, TypeVar, cast, Tuple
from collections.abc import Callable
#import traceback
import logging
import colorsys


LOGGER = logging.getLogger(__name__)

# EFFECT_0x03_0x00 = "Colorloop"
# EFFECT_0x03_0x01 = "Red fade"
# EFFECT_0x03_0x02 = "Green fade"
# EFFECT_0x03_0x03 = "Blue fade"
# EFFECT_0x03_0x04 = "Yellow fade"
# EFFECT_0x03_0x05 = "Cyan fade"
# EFFECT_0x03_0x06 = "Magenta fade"
# EFFECT_0x03_0x07 = "White fade"
# EFFECT_0x03_0x08 = "Red green cross fade"
# EFFECT_0x03_0x09 = "Red blue cross fade"
# EFFECT_0x03_0x0a = "Green blue cross fade"
# EFFECT_0x03_0x0b = "effect_0x03_0x0b"
# EFFECT_0x03_0x0c = "Color strobe"
# EFFECT_0x03_0x0d = "Red strobe"
# EFFECT_0x03_0x0e = "Green strobe"
# EFFECT_0x03_0x0f = "Blue strobe"
# EFFECT_0x03_0x10 = "Yellow strobe"
# EFFECT_0x03_0x11 = "Cyan strobe"
# EFFECT_0x03_0x12 = "Magenta strobe"
# EFFECT_0x03_0x13 = "White strobe"
# EFFECT_0x03_0x14 = "Color jump"
# EFFECT_0x03_0x15 = "RGB jump"


# EFFECT_MAP = {
#     EFFECT_0x03_0x00:    (0x03,0x00),
#     EFFECT_0x03_0x01:    (0x03,0x01),
#     EFFECT_0x03_0x02:    (0x03,0x02),
#     EFFECT_0x03_0x03:    (0x03,0x03),
#     EFFECT_0x03_0x04:    (0x03,0x04),
#     EFFECT_0x03_0x05:    (0x03,0x05),
#     EFFECT_0x03_0x06:    (0x03,0x06),
#     EFFECT_0x03_0x07:    (0x03,0x07),
#     EFFECT_0x03_0x08:    (0x03,0x08),
#     EFFECT_0x03_0x09:    (0x03,0x09),
#     EFFECT_0x03_0x0a:    (0x03,0x0a),
#     EFFECT_0x03_0x0b:    (0x03,0x0b),
#     EFFECT_0x03_0x0c:    (0x03,0x0c),
#     EFFECT_0x03_0x0d:    (0x03,0x0d),
#     EFFECT_0x03_0x0e:    (0x03,0x0e),
#     EFFECT_0x03_0x0f:    (0x03,0x0f),
#     EFFECT_0x03_0x10:    (0x03,0x10),
#     EFFECT_0x03_0x11:    (0x03,0x11),
#     EFFECT_0x03_0x12:    (0x03,0x12),
#     EFFECT_0x03_0x13:    (0x03,0x13),
#     EFFECT_0x03_0x14:    (0x03,0x14),
#     EFFECT_0x03_0x15:    (0x03,0x15)
# }

EFFECT_LIST = [e.value for e in Effect]
# EFFECT_ID_NAME = {v: k for k, v in EFFECT_MAP.items()}

DEFAULT_ATTEMPTS = 3
BLEAK_BACKOFF_TIME = 0.25
RETRY_BACKOFF_EXCEPTIONS = (BleakDBusError)
RECONNECT_BASE_DELAY = 1
RECONNECT_MAX_DELAY = 30
RECONNECT_CONNECT_TIMEOUT = 6
RECONNECT_EXCEPTIONS = (ConnectionError, BleakNotFoundError, *BLEAK_EXCEPTIONS)
UNEXPECTED_DISCONNECT_LOG_INTERVAL = 60

WrapFuncType = TypeVar("WrapFuncType", bound=Callable[..., Any])

def retry_bluetooth_connection_error(func: WrapFuncType) -> WrapFuncType:
    async def _async_wrap_retry_bluetooth_connection_error(
        self: "BJLEDInstance", *args: Any, **kwargs: Any
    ) -> Any:
        attempts = DEFAULT_ATTEMPTS
        max_attempts = attempts - 1

        for attempt in range(attempts):
            try:
                return await func(self, *args, **kwargs)
            except BleakNotFoundError:
                # The lock cannot be found so there is no
                # point in retrying.
                raise
            except RETRY_BACKOFF_EXCEPTIONS as err:
                if attempt >= max_attempts:
                    LOGGER.debug(
                        "%s: %s error calling %s, reach max attempts (%s/%s)",
                        self.name,
                        type(err),
                        func,
                        attempt,
                        max_attempts,
                        exc_info=True,
                    )
                    raise
                LOGGER.debug(
                    "%s: %s error calling %s, backing off %ss, retrying (%s/%s)...",
                    self.name,
                    type(err),
                    func,
                    BLEAK_BACKOFF_TIME,
                    attempt,
                    max_attempts,
                    exc_info=True,
                )
                await asyncio.sleep(BLEAK_BACKOFF_TIME)
            except BLEAK_EXCEPTIONS as err:
                if attempt >= max_attempts:
                    LOGGER.debug(
                        "%s: %s error calling %s, reach max attempts (%s/%s): %s",
                        self.name,
                        type(err),
                        func,
                        attempt,
                        max_attempts,
                        err,
                        exc_info=True,
                    )
                    raise
                LOGGER.debug(
                    "%s: %s error calling %s, retrying  (%s/%s)...: %s",
                    self.name,
                    type(err),
                    func,
                    attempt,
                    max_attempts,
                    err,
                    exc_info=True,
                )
                await asyncio.sleep(BLEAK_BACKOFF_TIME)

    return cast(WrapFuncType, _async_wrap_retry_bluetooth_connection_error)


class BJLEDInstance:
    def __init__(self, address, reset: bool, delay: int, hass) -> None:
        self.loop = asyncio.get_running_loop()
        self._mac = address
        self._reset = reset
        try:
            self._delay = max(0, int(delay))
        except (TypeError, ValueError):
            self._delay = 0
        self._hass = hass
        self._device: BLEDevice | None = None
        self._device = bluetooth.async_ble_device_from_address(self._hass, address)
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        self._client: BleakClientWithServiceCache | None = None
        self._disconnect_timer: asyncio.TimerHandle | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._cached_services: BleakGATTServiceCollection | None = None
        self._expected_disconnect = False
        self._last_unexpected_disconnect_log = 0.0
        self._is_on = False
        self._rgb_color = None
        self._brightness = 254
        self._effect = None
        self._effect_speed = 0x1
        self._color_mode = ColorMode.RGB
        self._turn_on_cmd = None
        self._turn_off_cmd = None
        self._compiler: StateCompiler | None = None
        self._state: LightState = self._initial_state()
        self._model = None
        self._transmitter: Transmitter | None = None
        if self._device:
            self._model = self._detect_model(self._device)
        LOGGER.debug(
            "Keepsmile instance initialized. Device discovered=%s MAC=%s",
            bool(self._device),
            self._mac,
        )

    @staticmethod
    def _initial_state():
        state = LightState()
        state.update(SwitchCommand(on=True))
        state.update(BrightnessCommand(0xfe))
        state.update(RGBCommand(0x5f, 0x0f, 0x40))

        return state

    def _detect_model(self, device: BLEDevice):
        profile = device_profile_from_ble_device(device)
        if profile is None:
            LOGGER.debug(
                "Bluetooth device profile not recognized yet: %s (%s)",
                device.name,
                device.address,
            )
            return None

        LOGGER.debug(
            "Bluetooth device recognized: %s. MAC: %s",
                device.name,
                device.address
        )

        self._compiler = profile.compiler()
        
        if RGBCommand in profile.supported_commands:
            self._color_mode = ColorMode.RGB
        elif BrightnessCommand in profile.supported_commands:
            self._color_mode = ColorMode.BRIGHTNESS
        else:
            self._color_mode = ColorMode.ONOFF
        
        return profile

    def _ensure_profile(self) -> None:
        """Ensure that a supported model profile has been detected."""
        if self._model is not None and self._compiler is not None:
            return
        if self._device is None:
            raise ConnectionError(f"{self._mac}: device not available in bluetooth scanner")
        profile = self._detect_model(self._device)
        if profile is None or self._compiler is None:
            raise ConnectionError(
                f"{self._mac}: unable to identify supported profile for device"
            )
        self._model = profile

    def restore_state(
        self,
        *,
        is_on: bool | None = None,
        brightness: int | None = None,
        rgb_color: Tuple[int, int, int] | None = None,
        effect: str | None = None,
    ) -> None:
        """Restore optimistic state values from Home Assistant's state cache."""
        if is_on is not None:
            self._is_on = is_on
            self._state.update(SwitchCommand(on=is_on))
        if brightness is not None:
            self._brightness = int(brightness)
            self._state.update(BrightnessCommand(self._brightness))
        if rgb_color is not None and len(rgb_color) == 3:
            self._rgb_color = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))
            self._state.update(RGBCommand(*self._rgb_color))
        if effect and effect in EFFECT_LIST:
            self._effect = effect
            self._state.update(EffectCommand(Effect(effect)))

    async def warmup(self) -> None:
        """Try a non-blocking startup connection so first command is faster."""
        try:
            await asyncio.wait_for(self._ensure_connected(), timeout=15)
            LOGGER.debug("%s: Startup warmup successful", self.name)
        except Exception as err:
            LOGGER.debug("%s: Startup warmup deferred: %s", self.name, err)
            
    async def _write_state(self):
        """Sends commands to the device so its configuration matches 
        the desired state, self._state"""
        platform_commands = self._compiler.compile(self._state)
        LOGGER.debug(f"Sending commands to {self.name}: {platform_commands}")
        await self._ensure_connected()
        if self._transmitter is None:
            raise ConnectionError(f"{self.name}: no transmitter available after connect")
        await self._transmitter.send_all(platform_commands)

    @property
    def mac(self):
        return self._mac

    @property
    def reset(self):
        return self._reset

    @property
    def name(self):
        if self._device and self._device.name:
            return self._device.name
        return self._mac

    @property
    def rssi(self):
        if self._device:
            return self._device.rssi
        return None

    @property
    def is_on(self):
        return self._is_on

    @property
    def brightness(self):
        return self._brightness 

    @property
    def rgb_color(self):
        return self._rgb_color

    @property
    def effect_list(self) -> list[str]:
        return EFFECT_LIST

    @property
    def effect(self):
        return self._effect
    
    @property
    def color_mode(self):
        return self._color_mode

    @retry_bluetooth_connection_error
    async def set_rgb_color(self, rgb: Tuple[int, int, int], brightness: int | None = None):
        self._rgb_color = rgb
        if brightness is None:
            brightness = self._brightness if self._brightness is not None else 254
        self._brightness = brightness
        # RGB packet
        self._state.update(RGBCommand(*rgb))
        self._state.update(BrightnessCommand(brightness))
        await self._write_state()

    @retry_bluetooth_connection_error
    async def set_brightness_local(self, brightness: int):
        # 0 - 254, should convert automatically with the hex calls
        # call color temp or rgb functions to update
        self._brightness = brightness
        self._state.update(BrightnessCommand(brightness))
        await self._write_state()
        # await self.set_rgb_color(self._rgb_color, value)

    @retry_bluetooth_connection_error
    async def turn_on(self):
        self._state.update(SwitchCommand(on=True))
        await self._write_state()
        self._is_on = True
                
    @retry_bluetooth_connection_error
    async def turn_off(self):
        self._state.update(SwitchCommand(on=False))
        await self._write_state()
        self._is_on = False

    @retry_bluetooth_connection_error
    async def set_effect(self, effect_name: str):
        if effect_name not in EFFECT_LIST:
            LOGGER.error("Effect %s not supported", effect_name)
            return
        self._effect = effect_name
        effect = Effect(effect_name)

        LOGGER.debug('Effect: %s', effect)
        
        self._state.update(EffectCommand(effect))
        await self._write_state()

    # @retry_bluetooth_connection_error
    # async def turn_on(self):
    #     await self._write(self._turn_on_cmd)
    #     self._is_on = True

    # @retry_bluetooth_connection_error
    # async def turn_off(self):
    #     await self._write(self._turn_off_cmd)
    #     self._is_on = False

    @retry_bluetooth_connection_error
    async def update(self):
        LOGGER.debug("%s: Update in bjled called", self.name)
        # I dont think we have anything to update

    async def _ensure_connected(self) -> None:
        """Ensure connection to device is established."""
        if self._connect_lock.locked():
            LOGGER.debug(
                "%s: Connection already in progress, waiting for it to complete",
                self.name,
            )
        if self._client and self._client.is_connected:
            self._reset_disconnect_timer()
            return
        async with self._connect_lock:
            # Check again while holding the lock
            if self._client and self._client.is_connected:
                self._reset_disconnect_timer()
                return

            current_ble_device = bluetooth.async_ble_device_from_address(
                self._hass, self._mac
            )
            if current_ble_device:
                self._device = current_ble_device
            elif self._device is not None:
                LOGGER.debug(
                    "%s: Scanner has no fresh BLE advertisement, using cached BLE device",
                    self.name,
                )
            else:
                raise ConnectionError(f"{self._mac}: bluetooth device is not currently discovered")
            self._ensure_profile()

            LOGGER.debug("%s: Connecting", self.name)
            client = await establish_connection(
                BleakClientWithServiceCache,
                self._device,
                self.name,
                self._disconnected,
                cached_services=self._cached_services,
                ble_device_callback=lambda: self._device,
            )
            LOGGER.debug("%s: Connected", self.name)

            self._cached_services = None
            self._transmitter = None
            transmitter: Transmitter | None = None
            model = self._model
            if model is None:
                raise ConnectionError(f"{self._mac}: supported profile not ready")
            try:
                transmitter = model.get_transmitter(client)
            except ConnectionError:
                LOGGER.debug("Connection failed: failed to wrap client with transmitter", exc_info=True)

                # Try to handle services failing to load
                try:
                    services = await client.get_services()
                    LOGGER.debug(f"Tried reloading characteristrics: {services.characteristics}")
                    transmitter = model.get_transmitter(client)

                except ConnectionError as err:
                    LOGGER.debug("Connection failed (x2): failed to wrap client with transmitter", exc_info=True)
                    with contextlib.suppress(Exception):
                        await client.disconnect()
                    raise ConnectionError("failed to initialize BLE transmitter") from err

            self._cached_services = client.services

            self._client = client
            self._transmitter = transmitter
            self._reconnect_task = None
            self._reset_disconnect_timer()

    def _reset_disconnect_timer(self) -> None:
        """Reset disconnect timer."""
        if self._disconnect_timer:
            self._disconnect_timer.cancel()
            self._disconnect_timer = None

        self._expected_disconnect = False
        if self._delay > 0:
            LOGGER.debug(
                "%s: Configured disconnect from device in %s seconds",
                self.name,
                self._delay
            )
            self._disconnect_timer = self.loop.call_later(self._delay, self._disconnect)
        else:
            LOGGER.debug("%s: Persistent connection enabled (no idle disconnect)", self.name)

    def _disconnected(self, client: BleakClientWithServiceCache) -> None:
        """Disconnected callback."""
        # Bleak can still invoke callbacks for an old client after a reconnect.
        # Ignore stale client disconnect events so we don't drop a healthy session.
        if self._client is not None and client is not self._client:
            LOGGER.debug("%s: Ignoring stale client disconnect callback", self.name)
            return

        if self._disconnect_timer:
            self._disconnect_timer.cancel()
            self._disconnect_timer = None

        self._client = None
        self._transmitter = None

        if self._expected_disconnect:
            LOGGER.debug("%s: Disconnected from device", self.name)
            self._expected_disconnect = False
            return

        now = time.monotonic()
        if now - self._last_unexpected_disconnect_log >= UNEXPECTED_DISCONNECT_LOG_INTERVAL:
            LOGGER.warning("%s: Device unexpectedly disconnected", self.name)
            self._last_unexpected_disconnect_log = now
        else:
            LOGGER.debug("%s: Device unexpectedly disconnected", self.name)
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Schedule a background reconnect when persistent mode is enabled."""
        if self._delay > 0:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = self.loop.create_task(self._reconnect_after_disconnect())

    async def _reconnect_after_disconnect(self) -> None:
        """Try reconnecting in the background after an unexpected disconnect."""
        attempt = 0
        while self._delay == 0:
            attempt += 1
            try:
                await asyncio.sleep(min(RECONNECT_BASE_DELAY * attempt, RECONNECT_MAX_DELAY))
                await asyncio.wait_for(self._ensure_connected(), timeout=RECONNECT_CONNECT_TIMEOUT)
                LOGGER.debug("%s: Background reconnect successful", self.name)
                return
            except (RECONNECT_EXCEPTIONS, TimeoutError) as err:
                LOGGER.debug(
                    "%s: Background reconnect attempt %s failed: %s",
                    self.name,
                    attempt,
                    err,
                    exc_info=True,
                )
        LOGGER.debug("%s: Background reconnect stopped (non-persistent mode)", self.name)

    def _disconnect(self) -> None:
        """Disconnect from device."""
        self._disconnect_timer = None
        asyncio.create_task(self._execute_timed_disconnect())

    async def stop(self) -> None:
        """Stop the LEDBLE."""
        LOGGER.debug("%s: Stop", self.name)
        if self._disconnect_timer:
            self._disconnect_timer.cancel()
            self._disconnect_timer = None
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconnect_task
            self._reconnect_task = None
        await self._execute_disconnect()

    async def _execute_timed_disconnect(self) -> None:
        """Execute timed disconnection."""
        LOGGER.debug(
            "%s: Disconnecting after timeout of %s",
            self.name,
            self._delay
        )
        await self._execute_disconnect()

    async def _execute_disconnect(self) -> None:
        """Execute disconnection."""
        async with self._connect_lock:
            client = self._client
            transmitter = self._transmitter
            self._expected_disconnect = True
            self._client = None
            self._transmitter = None
            if client and client.is_connected:
                if transmitter:
                    # Calls client.disconnect internally
                    await transmitter.close()
                else:
                    await client.disconnect()
            LOGGER.debug("%s: Disconnected", self.name)
    
